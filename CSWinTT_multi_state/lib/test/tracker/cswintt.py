from lib.test.tracker.basetracker import BaseTracker
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
import os
from lib.models import build_cswintt
from lib.test.tracker.cswintt_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.image import *


def map_box_back(state, pred_box: list, resize_factor: float):
	if len(state) == 1:
		state = state[0]
	cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
	cx, cy, w, h = pred_box
	half_side = 0.5 * 384 / resize_factor
	cx_real = cx + (cx_prev - half_side)
	cy_real = cy + (cy_prev - half_side)
	return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

class CSWinTT(BaseTracker):
	def __init__(self, params, dataset_name):
		super(CSWinTT, self).__init__(params)
		network = build_cswintt(params.cfg)
		network.load_state_dict(torch.load(self.params.checkpoint_cls, map_location='cpu'), strict=False)
		print("load----" + self.params.checkpoint_cls)
		self.cfg = params.cfg
		self.network = network.cuda()
		self.network.eval()
		self.preprocessor = Preprocessor()
		# self.state = None
		self.state = []
		# for debug
		self.debug = False
		self.frame_id = 0
		if self.debug:
			self.save_dir = "debug"
			if not os.path.exists(self.save_dir):
				os.makedirs(self.save_dir)
		# for save boxes from all queries
		self.save_all_boxes = params.save_all_boxes
		# template update
		self.z_dict1 = {}
		self.z_dict_list = []

		self.max_confidence = 0
		self.max_template = [0, 0, 0, 0]
		self.last_sequence = False
		self.max_pred_box = [0, 0, 0, 0]

		self.focal_max_template = []
		self.focal_max_template_feature = []
		self.first_track = True
		self.conf = []
		# Set the update interval
		DATASET_NAME = dataset_name.upper()
		if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
			self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
		else:
			self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
		print("Update interval is: ", self.update_intervals)
		self.num_extra_template = len(self.update_intervals)

	def initialize(self, image, info: dict):
		# initialize z_dict_list
		self.z_dict_list = []
		# get the 1st template
		z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
		                                              output_sz=self.params.template_size)  # template_size: 128
		# 전체 영상 보지말고 주어진 영역의 384, 384 크기로만 봐라.
		template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1) # tensor로 만들어주는 애
		with torch.no_grad():
			self.z_dict1 = self.network.forward_backbone(template1)  # 8,8,256
		# get the complete z_dict_list, (two templates: fixed template + update template(latest high conf-score))
		self.z_dict_list.append(self.z_dict1)
		for i in range(self.num_extra_template):
			self.z_dict_list.append([deepcopy(self.z_dict1)])

		# save states
		# self.state = info['init_bbox']  # one_state tracking
		self.state.append([info['init_bbox']])  # multi_state tracking
		self.frame_id = 0
		if self.save_all_boxes:
			'''save all predicted boxes'''
			all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
			return {"all_boxes": all_boxes_save}

	def track_multi_state(self, image, info):
		H, W, _ = image.shape
		self.frame_id += 1

		self.max_confidence = 0  # 추적 결과에서 가장 높은 점수를 저장
		self.max_resize_factor = 0  # 가장 높은 점수에서의 resize factor 를 저장
		for state_idx, state in enumerate(self.state):  # 복수개의 state 로 부터 하나씩 search region 을 뽑아 추적한다.
			# get the search region
			if len(state) > 1:
				state_info = state[1]
			state = state[0]
			x_patch_arr, resize_factor, x_amask_arr = sample_target(image, state, self.params.search_factor,
																	output_sz=self.params.search_size)  # search_size:384 # (x1, y1, w, h)
			search = self.preprocessor.process(x_patch_arr, x_amask_arr)

			with torch.no_grad():
				x_dict = self.network.forward_backbone(search)  # search feature 를 뽑아낸다.
				feat_dict_list = [self.z_dict_list[0]] + [self.z_dict_list[1][state_idx]] + [x_dict]  # fixed template + online updated template + search feature
				out_dict, _, _ = self.network.forward_transformer(feat_dict_list, run_box_head=True, run_cls_head=True)  # 추적

			# get the final result
			pred_boxes = out_dict['pred_boxes'].view(-1, 4)
			# Baseline: Take the mean of all pred boxes as the final result
			pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
			conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

			if self.max_confidence < conf_score:  # max 값들 업데이트
				self.max_confidence = conf_score
				self.max_state = state
				self.max_pred_box = pred_box
				self.max_resize_factor = resize_factor

		# clip_box(map_box_back(state, pred_box, resize_factor), H, W, margin=10)
		# 복수개의 state 중에서 가장 점수가 높았던 결과를 저장함
		max_state = clip_box(map_box_back(self.max_state, self.max_pred_box, self.max_resize_factor), H, W, margin=10)
		z_patch_arr, _, z_amask_arr = sample_target(image, max_state, self.params.template_factor,
													output_sz=self.params.template_size)  # (x1, y1, w, h)
		template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)

		with torch.no_grad():
			z_dict_t = self.network.forward_backbone(template_t)
		if self.first_track and self.max_confidence > 0.5:  # 첫번째 프레임에서 추적할때 0.5 보다 높으면 template 에 추가함.
			self.focal_max_template.append([self.max_confidence, max_state, z_dict_t, info])
			# self.focal_max_template_feature.append(z_dict_t)
		# elif not self.first_track:  # 두번째 프레임에서부터는 점수 상관없이 모두 추가함
		elif not self.first_track and self.max_confidence > 0.5:  # 두번째 프레임에서부터는 점수 상관없이 모두 추가함
			self.focal_max_template.append([self.max_confidence, max_state, z_dict_t, info])

		if self.last_sequence:  # 마지막 포컬 시퀀스를 추적할때
			# confidence 점수 기준으로 정렬
			if not self.first_track:
				# state, online update template 업데이트
				# self.focal_max_template = sorted(self.focal_max_template, key=lambda x: -x[0])[:6]  # 내림차순
				# self.focal_max_template = sorted(self.focal_max_template, key=lambda x: x[0])[5:]  # 오름차순
				self.focal_max_template = sorted(self.focal_max_template, key=lambda x: x[0])[-5:]  # 오름차순

				state_len = len(self.conf)
				cnt = 0
				for conf, state, template, info in self.focal_max_template:
					if conf > 0.5:
						cnt += 1
						if state_len >= cnt:
							self.conf.pop()
							self.state.pop()
							self.z_dict_list[1].pop()
						self.conf.insert(0, conf)
						self.state.insert(0, [state, info])
						self.z_dict_list[1].insert(0, template)
			else:
				# first tracking
				self.state = []
				self.z_dict_list[1] = []

				self.focal_max_template = sorted(self.focal_max_template, key=lambda x: x[0])[-5:]
				for conf, state, template, info in self.focal_max_template:
					self.conf.append(conf)
					self.state.append([state, info])
					self.z_dict_list[1].append(template)
			self.last_sequence = False
			self.focal_max_template = []

		return {"target_bbox": max_state, "conf_score": self.max_confidence}


	def track_one_state(self, image, info: dict = None):
		H, W, _ = image.shape
		self.frame_id += 1

		# self.max_confidence = 0  # 추적 결과에서 가장 높은 점수를 저장
		# self.max_resize_factor = 0  # 가장 높은 점수에서의 resize factor 를 저장

		# for state_idx, state in enumerate(self.state):  # 복수개의 state 로 부터 하나씩 search region 을 뽑아 추적한다.
			# get the search region
		if len(self.state) == 1:
			self.state = self.state[0]
		x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
																	output_sz=self.params.search_size)  # search_size:384 # (x1, y1, w, h)
		search = self.preprocessor.process(x_patch_arr, x_amask_arr)

		with torch.no_grad():
			x_dict = self.network.forward_backbone(search)  # search feature 를 뽑아낸다.
			feat_dict_list = [self.z_dict_list[0]] + [self.z_dict_list[1][0]] + [x_dict]  # fixed template + online updated template + search feature
			out_dict, _, _ = self.network.forward_transformer(feat_dict_list, run_box_head=True, run_cls_head=True)  # 추적

			# get the final result
		pred_boxes = out_dict['pred_boxes'].view(-1, 4)
			# Baseline: Take the mean of all pred boxes as the final result
		pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
		conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()

		state = clip_box(map_box_back(self.state, pred_box, resize_factor), H, W, margin=10)

		# max 값들 업데이트
		if self.max_confidence < conf_score:
			self.max_confidence = conf_score
			self.max_state = state
			self.max_pred_box = pred_box
			self.max_resize_factor = resize_factor

			z_patch_arr, _, z_amask_arr = sample_target(image, self.max_state, self.params.template_factor,
														output_sz=self.params.template_size)  # (x1, y1, w, h)
			self.max_template = self.preprocessor.process(z_patch_arr, z_amask_arr)
			# with torch.no_grad():
			# 	self.z_dict_t = self.network.forward_backbone(template_t)

		# update template
		if self.last_sequence:  # 마지막 포컬 시퀀스를 추적할때
			self.state = self.max_state
			if self.max_confidence > 0.5:
				# z_patch_arr, _, z_amask_arr = sample_target(image, self.max_state, self.params.template_factor,
				# 											output_sz=self.params.template_size)  # (x1, y1, w, h)
				# template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
				with torch.no_grad():
					self.z_dict_list[1][0] = self.network.forward_backbone(self.max_template)
			self.max_confidence = 0
			self.last_sequence = False
			# if not self.first_track:
			# 	# state, online update template 업데이트
			# 	# self.focal_max_template = sorted(self.focal_max_template, key=lambda x: -x[0])[:6]  # 내림차순
			# 	self.focal_max_template = sorted(self.focal_max_template, key=lambda x: x[0])[5:]  # 오름차순
			# 	state_len = len(self.conf)
			# 	cnt = 0
			# 	for conf, state, template in self.focal_max_template:
			# 		if conf > 0.5:
			# 			cnt += 1
			# 			if state_len > cnt:
			# 				self.conf.pop()
			# 				self.state.pop()
			# 				self.z_dict_list[1].pop()
			# 			self.conf.insert(0, conf)
			# 			self.state.insert(0, state)
			# 			self.z_dict_list[1].insert(0, template)
			# else:
			# 	# first tracking
			# 	self.state = []
			# 	self.z_dict_list[1] = []
			#
			# 	self.focal_max_template = sorted(self.focal_max_template, key=lambda x: -x[0])
			# 	for conf, state, template in self.focal_max_template:
			# 		self.conf.append(conf)
			# 		self.state.append(state)
			# 		self.z_dict_list[1].append(template)
			# self.last_sequence = False
			# self.focal_max_template = []

		return {"target_bbox": state, "conf_score": conf_score}


	def map_box_back(self, pred_box: list, resize_factor: float):
		cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
		cx, cy, w, h = pred_box
		half_side = 0.5 * self.params.search_size / resize_factor
		cx_real = cx + (cx_prev - half_side)
		cy_real = cy + (cy_prev - half_side)
		return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

	def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
		cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
		cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
		half_side = 0.5 * self.params.search_size / resize_factor
		cx_real = cx + (cx_prev - half_side)
		cy_real = cy + (cy_prev - half_side)
		return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

def get_tracker_class():
	return CSWinTT
