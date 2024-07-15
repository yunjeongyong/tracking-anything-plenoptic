import importlib
import os
import torch
from CSWinTT_multi_state.demo.utils.utils.metric import pytorch_iou
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
import CSWinTT_multi_state.demo.dataloaders.video_transforms as tr
# from lib.test.evaluation.environment import env_settings
import time
from CSWinTT_multi_state.demo.segment_anything_hq.segment_anything.segment_anything.modeling.mask_decoder import MaskDecoder
from torchvision import transforms
import cv2
import cv2 as cv
import csv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import time
import random
from CSWinTT_multi_state.demo.segment_anything_hq.segment_anything.segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor, build_sam, build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from CSWinTT_multi_state.demo.networks.models import build_vos_model
from CSWinTT_multi_state.demo.networks.engines import build_engine
from CSWinTT_multi_state.demo.utils.utils.checkpoint import load_network
from CSWinTT_multi_state.demo.utils.utils.image import flip_tensor
# from CSWinTT_multi_state.demo.segment_anything.segment_anything import build_sam

from CSWinTT_multi_state.demo.segment_anything_hq.segment_anything.segment_anything.modeling import ImageEncoderViT, MaskDecoderHQ, PromptEncoder, Sam, TwoWayTransformer
import os
import torch
import gc
#https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
import gc
gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512' #'garbage_collection_threshold:0.8,max_split_size_mb:512'

from torch.nn import DataParallel


def csv_write(save_path, bbox, save_name):
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f'{save_name}.csv'
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(bbox)
    f.close()


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1000000007)
torch.set_num_threads(4)
torch.autograd.set_grad_enabled(False)

cur_colors = [(0, 255, 255), # yellow b g r
              (255, 0, 0), # blue
              (0, 255, 0), # green
              (0, 0, 255), # red
              (255, 255, 255), # white
              (0, 0, 0), # black
              (255, 255, 0), # Cyan
              (225, 228, 255), # MistyRose
              (180, 105, 255), # HotPink
              (255, 0, 255), # Magenta
              ]*100

class AOTTracker(object):
    def __init__(self, cfg, gpu_id):
        self.with_crop = False
        self.EXPAND_SCALE = None
        self.small_ratio = 12
        self.mid_ratio = 100
        self.large_ratio = 0.5
        self.AOT_INPUT_SIZE = (465, 465)
        self.cnt = 2
        self.gpu_id = gpu_id
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(gpu_id)
        self.model.cuda(gpu_id)
        self.model.eval()
        print('cfg.TEST_CKPT_PATH = ', cfg.TEST_CKPT_PATH)
        self.model, _ = load_network(self.model, cfg.TEST_CKPT_PATH, gpu_id)
        self.aug_nums = len(cfg.TEST_MULTISCALE)
        if cfg.TEST_FLIP:
            self.aug_nums *= 2
        self.engine = []
        for aug_idx in range(self.aug_nums):
            self.engine.append(build_engine(cfg.MODEL_ENGINE,
                                            phase='eval',
                                            aot_model=self.model,
                                            gpu_id=gpu_id,
                                            short_term_mem_skip=cfg.TEST_SHORT_TERM_MEM_SKIP,
                                            long_term_mem_gap=cfg.TEST_LONG_TERM_MEM_GAP,
                                            ))
            self.engine[-1].eval()
        self.transform = transforms.Compose([
            tr.MultiRestrictSize_(cfg.TEST_MAX_SHORT_EDGE,
                                  cfg.TEST_MAX_LONG_EDGE, cfg.TEST_FLIP, cfg.TEST_INPLACE_FLIP,
                                  cfg.TEST_MULTISCALE, cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

    def add_first_frame(self, frame, mask):
        sample = {
            'current_img': frame,
            'current_label': mask,
            'height': frame.shape[0],
            'weight': frame.shape[1]
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        for aug_idx in range(self.aug_nums):
            frame = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = sample[aug_idx]['current_label'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            mask = F.interpolate(mask, size=frame.size()[2:], mode="nearest")
            self.engine[aug_idx].add_reference_frame(frame, mask, frame_step=0, obj_nums=int(mask.max()))

    def track(self, image):

        height = image.shape[0]
        width = image.shape[1]
        sample = {'current_img': image}
        sample['meta'] = {
            'height': height,
            'width': width,
            'flip': False
        }
        sample = self.transform(sample)

        if self.aug_nums > 1:
            torch.cuda.empty_cache()
        all_preds = []
        for aug_idx in range(self.aug_nums):
            output_height = sample[aug_idx]['meta']['height']
            output_width = sample[aug_idx]['meta']['width']
            image = sample[aug_idx]['current_img'].unsqueeze(0).float().cuda(self.gpu_id, non_blocking=True)
            image = image.cuda(self.gpu_id, non_blocking=True)
            self.engine[aug_idx].match_propogate_one_frame(image)
            is_flipped = sample[aug_idx]['meta']['flip']
            pred_logit = self.engine[aug_idx].decode_current_logits((output_height, output_width))
            if is_flipped:
                pred_logit = flip_tensor(pred_logit, 3)
            pred_prob = torch.softmax(pred_logit, dim=1)
            all_preds.append(pred_prob)
            cat_all_preds = torch.cat(all_preds, dim=0)
            pred_prob = torch.mean(cat_all_preds, dim=0, keepdim=True)
            pred_label = torch.argmax(pred_prob, dim=1, keepdim=True).float()
            _pred_label = F.interpolate(pred_label,
                                        size=self.engine[aug_idx].input_size_2d,
                                        mode="nearest")
            self.engine[aug_idx].update_memory(_pred_label)
            mask = pred_label.detach().cpu().numpy()[0][0].astype(np.uint8)
        conf = 0

        return mask, conf

def read_img(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# def build_sam_vit_h(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=1280,
#         encoder_depth=32,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[7, 15, 23, 31],
#         checkpoint=checkpoint,
#     )
#
# def build_sam_vit_b(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_global_attn_indexes=[2, 5, 8, 11],
#         checkpoint=checkpoint,
#     )


SAM_prompt = 'Box' #'Box
build_sam = build_sam_vit_b
model_type = 'vit_b'
sam_checkpoint = "/media/mnt/Project/CSWinTT_multi_state/demo/pretrained_model/sam_hq_vit_b.pth"
output_mode = "binary_mask"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam = DataParallel(sam)
sam.to(device=torch.device('cuda:0'))
# sam.to(torch.device("cuda:1"), non_blocking=True)
# sam.to(device="cpu")
mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=1, output_mode=output_mode)
mask_prompt = SamPredictor(sam)


# def build_sam_vit_l(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=1024,
#         encoder_depth=24,
#         encoder_num_heads=16,
#         encoder_global_attn_indexes=[5, 11, 17, 23],
#         checkpoint=checkpoint,
#     )


# def build_sam_vit_b(checkpoint=None):
#     return _build_sam(
#         encoder_embed_dim=768,
#         encoder_depth=12,
#         encoder_num_heads=12,
#         encoder_global_attn_indexes=[2, 5, 8, 11],
#         checkpoint=checkpoint,
#     )

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

sam_model_registry = {
    "default": build_sam_vit_b,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

class HQTrack(object):
    def __init__(self, cfg, config, local_track=False,sam_refine=False,sam_refine_iou=0):
        self.mask_size = None
        self.local_track = local_track

        self.aot_tracker = AOTTracker(cfg, config['gpu_id'])
        # SAM
        self.sam_refine=sam_refine
        if self.sam_refine:
            model_type = 'vit_b' #'vit_h'
            sam_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'segment_anything/pretrained_model/sam_hq_vit_b.pth')
            output_mode = "binary_mask"
            print("1111111111111111", sam_checkpoint)
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=torch.device('cuda:0'))
            self.mask_generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)
            self.mask_prompt = SamPredictor(sam)
        self.sam_refine_iou=sam_refine_iou

        self.first_track = None
        self.last_sequence = False

    def get_box(self, label):
        thre = np.max(label) * 0.5
        label[label > thre] = 1
        label[label <= thre] = 0
        a = np.where(label != 0)
        height, width = label.shape
        ratio = 0.0

        if len(a[0]) != 0:
            bbox1 = np.stack([np.min(a[1]), np.min(a[0]), np.max(a[1]), np.max(a[0])])
            w, h = np.max(a[1]) - np.min(a[1]), np.max(a[0]) - np.min(a[0])
            x1 = max(bbox1[0] - w * ratio, 0)
            y1 = max(bbox1[1] - h * ratio, 0)
            x2 = min(bbox1[2] + w * ratio, width)
            y2 = min(bbox1[3] + h * ratio, height)
            bbox = np.array([x1, y1, x2, y2])
        else:
            bbox = np.array([0, 0, 0, 0])
        return bbox

    def initialize(self, image, mask):
        self.tracker = self.aot_tracker
        self.tracker.add_first_frame(image, mask)
        self.aot_mix_tracker = None
        self.mask_size = mask.shape

    def track(self, image):
        print("image_size", image.shape)
        m, confidence = self.tracker.track(image)
        print('confidence1', confidence)
        m = F.interpolate(torch.tensor(m)[None, None, :, :],
                          size=self.mask_size, mode="nearest").numpy().astype(np.uint8)[0][0]

        if self.sam_refine:
            obj_list = np.unique(m)
            mask_ = np.zeros_like(m)
            mask_2 = np.zeros_like(m)
            masks_ls = []
            for i in obj_list:
                mask = (m == i).astype(np.uint8)
                if i == 0 or mask.sum() == 0:
                    masks_ls.append(mask_)
                    continue
                bbox = self.get_box(mask)
                # box prompt
                self.mask_prompt.set_image(image)
                masks_, iou_predictions, _ = self.mask_prompt.predict(box=bbox)

                select_index = list(iou_predictions).index(max(iou_predictions))
                output = masks_[select_index].astype(np.uint8)
                iou = pytorch_iou(torch.from_numpy(output).cuda().unsqueeze(0),
                                  torch.from_numpy(mask).cuda().unsqueeze(0), [1])
                iou = iou.cpu().numpy()
                if iou < self.sam_refine_iou:
                    output = mask
                masks_ls.append(output)
                mask_2 = mask_2 + output * i
            masks_ls = np.stack(masks_ls)
            masks_ls_ = masks_ls.sum(0)
            masks_ls_argmax = np.argmax(masks_ls, axis=0)
            rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
            rs = np.array(rs).astype(np.uint8)
            print('confidence2', confidence)

            return rs, confidence
        return m, confidence


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        # env = env_settings()
        # if self.run_id is None:
        #     self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        # else:
        #     self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        # if result_only:
        #     self.results_dir = '{}/{}/{}'.format(env.results_path, "LaSOT", self.name)
        # print(os.path.join(os.path.dirname(__file__), '..', 'tracker', '%s.py' % self.name))
        # tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
        #                                                       '..', 'tracker', '%s.py' % self.name))
        # if os.path.isfile(tracker_module_abspath):
        #     tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
        #     self.tracker_class = tracker_module.get_tracker_class()
        # else:
        #     self.tracker_class = None

    def create_tracker(self, cfg, config, sam_refine, sam_refine_iou):
        # tracker = self.tracker_class(params, self.dataset_name)
        # return tracker

        tracker = HQTrack(cfg, config, True, sam_refine, sam_refine_iou)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            if frame_num == 80:
                print()

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
        i = 0
        while True:
            i += 1
            print(f'{i}th frame start!')
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.imwrite(f'result/{i:03d}.png', frame_disp)
            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def draw_bboxes(self, img, tracker_num, bbox, color, identities=None, offset=(0, 0)):
        x, y, w, h = bbox
        label = str(tracker_num)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]  # Nonvideo3
        print(img)
        print('x', x)
        print('y', y)
        print('h', h)
        print('w', w)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 5, 5)[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)  # Nonvideo3
        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)
        cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
        cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)  # Nonvideo3
        # cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 5, [255, 255, 255], 5)
        return img

    def run_focal(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the focal sequence."""
        from CSWinTT_multi_state.FocalDataloader import LoadFocalFolder
        from CSWinTT_multi_state.config_parser import ConfigParser
        import torch
        from datetime import datetime
        import sys

        config = ConfigParser('./config.json')
        write_bbox = config['write_bbox']  # bbox 좌표 저장할 것인지 여부
        write_conf = config['write_conf']  # confidence 값 저장할 것인지 여부
        write_time = config['write_time']  # 한 프레임당 추적 시간 저장할 것인지 여부
        is_record = config['is_record']  # 추적한 결과 저장할 것인지 여부
        video_name = config['video_name']  # 창의 이름
        start_frame_num = config['start_frame_num']  # 추적 시작할 프레임 번호
        last_frame_num = config['last_frame_num']  # 추적이 끝나는 프레임 번호
        start_focal_num = config['start_focal_num']  # 추적 시작할 포컬 번호
        last_focal_num = config['last_focal_num']  # 추적이 끝나는 포컬 번호
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')  # 시간
        save_path = Path(config['save_path']) / timestamp  # 현재 시간을 폴더 이름으로 지정
        save_path.mkdir(parents=True, exist_ok=True)  # 폴더 생성
        # params = self.get_parameters()
        cfg, engine_config, sam_refine, sam_refine_iou = self.get_parameters()

        # debug_ = debug
        # if debug is None:
        #     debug_ = getattr(params, 'debug', 0)
        # params.debug = debug_
        #
        # params.tracker_name = self.name
        # params.param_name = self.parameter_name

        # 사용되지 않음, 개선 필요
        output_boxes = []  # bbox 저장할 리스트

        # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

        # 데이터로더 초기화
        # 데이터로더에서 리턴하는 값은 (이미지(cv2), focal 경로들) 이다.
        focaldataloader = LoadFocalFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                         focal_range=(start_focal_num, last_focal_num))

        def _build_init_info(box):
            return {'init_bbox': box}

        frame_idx = start_frame_num
        bbox_color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]

        is_first_frame = True  # 현재 프레임이 첫번째 프레임인지 아닌지를 나타내는 변수
        print("Please type the number of trackers: ")
        tracker_num = int(sys.stdin.readline())
        # tracker = []
        # tracker: HQTrack = None
        tracker = self.create_tracker(cfg, engine_config, sam_refine, sam_refine_iou)
        # best_idx = [0 for _ in range(tracker_num)]  # 최고 점수가 기록될 리스트, 맨 처음에는 모두 0으로 초기화
        # d_index = [5 for _ in range(tracker_num)]  # d: 최적 포컬 인덱스에서 체크할 주변 인덱스 수, k-d ~ k+d (기본값 5)
        best_idx = 0
        d_index = 5
        baseline = False  # 사용되지 않음, 삭제 필요

        # for k in range(tracker_num):  # 추적할 객체의 수만큼 트래커 생성
        #     tracker.append(self.create_tracker(cfg, engine_config, sam_refine, sam_refine_iou))

        # frame: 이미지, focals: 경로 리스트
        for frame, focals in focaldataloader:
            frame_idx += 1
            img_ori = frame.copy()
            # if frame_idx > start_frame_num:  # 첫번째 프레임이 아닐 경우
            #     is_first_frame = False

            if is_first_frame:  # 첫번째 프레임일 경우
                # img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                is_first_frame = False
                masks_ls = []
                # print("mask_2", frame)
                mask_2 = np.zeros_like(img_ori[:, :, 0])
                masks_ls.append(mask_2)

                xywhs = [
                    [1307, 489, 259, 272],
                    [1483, 1975, 232, 124],
                ]
                for k in range(tracker_num):
                    # ROI를 입력받기 위한 준비
                    # cv2.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #            1.5, (0, 0, 0), 1)
                    # x, y, w, h = cv2.selectROI(video_name, frame, fromCenter=False)  # 마우스로 추적할 객체 지정
                    # x, y, w, h = 1307, 489, 259, 272
                    x, y, w, h = xywhs[k]
                    # init_state = [x, y, w, h]  # 입력한 박스를 리스트로 만들어서 init_state에 저장
                    # tracker[k].initialize(frame, _build_init_info(init_state))  # 트래커를 사용자가 마우스로 지정한 박스로 초기화
                    # output_boxes.append(init_state)

                    prompt = [x, y, x + w, y + h]
                    mask_prompt.set_image(img_ori)
                    print('mask_prompt', mask_prompt)
                    # if SAM_prompt == 'Box':
                    #     masks_, iou_predictions, _ = mask_prompt.predict(box=np.array(prompt).astype(float))
                    # elif SAM_prompt == 'Point':
                    #     masks_, iou_predictions, _ = mask_prompt.predict(point_labels=np.asarray([1]),
                    #                                                      point_coords=np.asarray([prompt]))
                    masks_, iou_predictions, _ = mask_prompt.predict(box=np.array(prompt).astype(float))
                    # print("masks_shape", masks_.shape)
                    select_index = list(iou_predictions).index(max(iou_predictions))
                    init_mask = masks_[select_index].astype(np.uint8)
                    # print('init_mask_shape', init_mask.shape)
                    # print('123123123', masks_, iou_predictions)
                    # print('mask_prompttttttt', mask_prompt)
                    # print('select_indexxxxx', select_index)
                    # masks_ls.append(init_mask)
                    masks_ls = np.append(masks_ls, init_mask)
                    #masks_ls.append(init_mask)
                    mask_2 = mask_2 + init_mask * (k + 1)

                    masks_ls = np.stack(masks_ls)
                    masks_ls_ = masks_ls.sum(0)
                    masks_ls_argmax = np.argmax(masks_ls, axis=0)
                    rs = np.where(masks_ls_ > 1, masks_ls_argmax, mask_2)
                    rs = np.array(rs).astype(np.uint8)
                    init_masks = []
                    for i in range(len(masks_ls)):
                        m_temp = rs.copy()
                        m_temp[m_temp != i + 1] = 0
                        m_temp[m_temp != 0] = 1
                        init_masks.append(m_temp)
                    print("init_maskssssssss", init_masks)

                    # img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_BGR2RGB)
                    img = img_ori.astype(np.float32)
                    for idx, m in enumerate(init_masks):
                        img[:, :, 1] += 127.0 * m
                        img[:, :, 2] += 127.0 * m
                        contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
                    im_m = im_m.clip(0, 255).astype(np.uint8)
                    cv2.putText(im_m, 'Init', (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)
                    #cv2.imshow(video_name, im_m)
                    #n = cv2.waitKey(1)
                    # HQtrack init
                    print('init target objects ...')
                    # tracker[k].initialize(img_ori, rs)
                    tracker.initialize(img_ori, rs)

            print(f'{frame_idx}th frame start!')
            if write_bbox:
                bbox_list = [frame_idx]
            if write_conf:
                conf_list = [frame_idx]
            if write_time:
                time_list = [frame_idx]

            # for k in range(tracker_num):
            if not is_first_frame:
                # 첫번째 프레임이 아닐 경우 트래커 객체의 first_track 변수 False로 설정
                tracker.first_track = False
            if write_time:
                # write_time 옵션이 True일 경우 시간 기록
                tracking_time_start = time.time()
            max_score = 0  # 최고 점수를 기록하기 위해 max_score 변수 0으로 초기화, 뒤에서 바뀔 예정
            best_state = [0, 0, 0, 0]  # 최고 점수에 해당하는 bbox가 저장될 예정
            if is_first_frame or baseline:
                # 첫번째 프레임일 때는 전체 focal 영역에서 탐색을 하기 때문에 range_focals를 focals로 초기화
                range_focals = focals
            elif d_index < best_idx < last_focal_num - d_index:
                # 찾으려는 범위가 (d_index ~ last_focal_num - d_index) 사이일 경우
                # 예를 들어서 d_index가 5고 last_focal_num이 100이면 (프레임당 focal이 100장이면)
                # best_idx가 5~95 사이이면 여기에 들어가게 된다.
                # 만약 best_idx가 30이고 d_index가 5면 [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] 이렇게 된다.
                range_focals = focals[best_idx - d_index: best_idx + d_index + 1]
                # Focal plane range 범위 벗어나지 않게 처리
            elif best_idx <= d_index:
                # 위의 예시에서 best_idx가 2라고 예를 들면 [-3 ~ 7] 이렇게 될텐데, 음수가 들어가면 안되니까
                # 강제로 [0 ~ 10]으로 조정해준다.
                range_focals = focals[0: d_index * 2 + 1]
            else:
                # 반대로 last_focal_num을 벗어나도 안되기 때문에,
                # 위의 예시에서 best_idx가 98이라고 한다면 [93 ~ 103]이라서 100을 넘어가게 되니까
                # 강제로 [90 ~ 100]으로 조정해준다.
                range_focals = focals[-(d_index * 2 + 1):]


            # if frame_idx > 180:
            #     a = frame_idx
            # focal sequence

            # img_ori = frame
            # m, confidence = tracker[k].track(img_ori)
            # print("mmmmmm", m)
            # print("confidencccccc", confidence)
            # pred_masks = []
            # for ii in range(tracker_num):
            #     m_temp = m.copy()
            #     m_temp[m_temp != ii + 1] = 0
            #     m_temp[m_temp != 0] = 1
            #     pred_masks.append(m_temp)
            # img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_BGR2RGB)
            # for idx, m in enumerate(pred_masks):
            #     img[:, :, 1] += 127.0 * m
            #     img[:, :, 2] += 127.0 * m
            #     contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #     im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
            # im_m = im_m.clip(0, 255).astype(np.uint8)
            # cv2.imshow(video_name, im_m)

            # range_focals의 개수는 d_index * 2 + 1
            # i는 0 ~ d_index*2
            # f는 각 focal들의 경로가 들어가게 된다.
            for i, f in enumerate(range_focals):
                if i == len(range_focals)-1:
                    # 루프의 마지막일 경우 트래커의 last_sequence를 True로 설정
                    tracker.last_sequence = True
                # f(경로명)를 파싱해서 숫자만 가져와서 i에 저장
                i = int(f.split('/')[-1][:3])

                # print('fffffff', f)
                # img = cv2.imread(f)  # 경로로부터 이미지 읽음
                # # img_ori = cv2.resize(img, (240, 135))
                img = frame
                img_ori = img.copy()

                # out = tracker[k].track(img)  # {"target_bbox": self.state, "conf_score": conf_score}
                # out = tracker[k].track_multi_state(img, f'{frame_idx}_{i}')  # {"target_bbox": self.state, "conf_score": conf_score}
                # def AOT():
                #     return mask, conf
                # def HQTrack():
                #     AOT
                #     return m, confidence
                # tracker = HQTrack()
                m, confidence = tracker.track(img_ori)
                print("mmmmmm", m)
                print("confidencccccc", confidence)
                # out = tracker[k].track_one_state(img)  # {"target_bbox": self.state, "conf_score": conf_score}
                # state = [int(s) for s in out['target_bbox']]  # out['target_bbox']와 동일한 배열 만듦 (단순히 state = out['target_bbox'] 로 써도 될듯)
                # score = float(out['conf'
                #                   '_score'])
                # print(f'[{k+1}th] focal {i} score: {score}')
                #
                # if max_score < score:  # 현재 focal이 최고 점수를 가질 때
                #     max_score = score  # max_score를 현재 점수로 변경
                #     output_boxes.append(state)
                #     best_idx[k] = int(f.split('\\')[-1][:3])  # best_idx를 현재 인덱스로 변경
                #     best_state = state  # best_state를 현재 bbox로 변경

                pred_masks = []
                for ii in range(tracker_num):
                    m_temp = m.copy()
                    m_temp[m_temp != ii + 1] = 0
                    m_temp[m_temp != 0] = 1
                    pred_masks.append(m_temp)
                # img = cv2.cvtColor(img_ori.astype(np.float32), cv2.COLOR_BGR2RGB)
                img = img_ori.astype(np.float32)
                for idx, m in enumerate(pred_masks):
                    img[:, :, 1] += 127.0 * m
                    img[:, :, 2] += 127.0 * m
                    contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    im_m = cv2.drawContours(img, contours, -1, cur_colors[idx], 2)
                im_m = im_m.clip(0, 255).astype(np.uint8)
                cv2.imshow(video_name, im_m)
                if is_record:
                    cv2.imwrite(f'{save_path}/{frame_idx:03d}.png', im_m)

            # self.draw_bboxes(frame, k + 1, best_state, bbox_color[k])  # 구해진 최고점수를 바탕으로 bbox 새로 그림
            # print(f'--[{k + 1}th] Best focal {best_idx[k]} score: {max_score}--')

            # d_index 조절
            if max_score > 0.5:  # 예측값이 좋으면 탐색 범위 축소 (3으로)
                d_index = 3
            else:
                d_index = 5  # 아니면 탐색 범위 늘림 (5로)
            # d_index[k] = 0
            if write_bbox:
                bbox_list.extend(best_state)
            if write_conf:
                conf_list.append(round(max_score, 2))
            if write_time:
                tracking_time = time.time() - tracking_time_start
                time_list.append(round(tracking_time, 2))
            # if is_record:
            #     cv2.imwrite(f'{save_path}/{frame_idx:03d}.png', frame)

            if write_bbox:
                csv_write(save_path, bbox_list, 'bbox')
            if write_conf:
                csv_write(save_path, conf_list, 'conf')
            if write_time:
                csv_write(save_path, time_list, 'time')
            # frame_idx += 1
            # print(f'best idx: {best_idx} focal num')
            cv2.imshow(video_name, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     exit()
            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        # cap.release()
        cv2.destroyAllWindows()

        # if save_results:
        #     if not os.path.exists(self.results_dir):
        #         os.makedirs(self.results_dir)
        #     # video_name = Path(videofilepath).stem
        #     base_results_path = 'result/result'
        #
        #     tracked_bb = np.array(output_boxes).astype(int)
        #     bbox_file = '{}.txt'.format(base_results_path)
        #     np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_2d(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        from ImagesDataloader import Load2DFolder
        from config_parser import ConfigParser
        import torch
        from datetime import datetime
        import cv2

        config = ConfigParser('./config.json')
        write_bbox = config['write_bbox']  # bbox 좌표 저장할 것인지 여부
        write_time = config['write_time']
        # write_gt = config['write_gt']
        is_record = config['is_record']
        video_name = config['video_name']
        video_type = config['video_type']
        img2d_ref = config['image2d_ref']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']
        # ckpt_path = config['pretrained_model']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_path = Path(config['save_path']) / timestamp
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        main_camera = 7
        load2DFolder = Load2DFolder("E:\\code\\SuperGluePretrainedNetwork-master\\unfold_images_color_@", main_camera, limit=None)
        first_frame_images = load2DFolder.__getitem__(0)
        print('first_frame_images', len(first_frame_images))
        # print('first_cameras', first_cameras)

        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

        def _build_init_info(box):
            return {'init_bbox': box}

        trackers = []
        tracker_colors = [list(np.random.random(size=3) * 256) for _ in range(len(first_frame_images))]
        print("first_frame_images_length:", len(first_frame_images))

        for i, image in enumerate(first_frame_images):
            x, y, w, h = cv2.selectROI(video_name, image, fromCenter=False)  # 마우스로 추적할 객체 지정
            init_state = [x, y, w, h]
            tracker = self.create_tracker(params)
            tracker.initialize(image, _build_init_info(init_state))
            out = tracker.track_one_state(image)  # {"target_bbox": self.state, "conf_score": conf_score}
            state = [int(s) for s in out['target_bbox']]
            score = round(out['conf_score'], 2)
            print(11111111, state)
            cv2.rectangle(image, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), tracker_colors[i], 1)
            print(222222222, str(first_cameras[i]))
            cv2.putText(image, str(first_cameras[i]), (state[0], state[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tracker_colors[i], 1)
            trackers.append(tracker)

        for idx, (frames, cameras) in enumerate(load2DFolder):
            print(f'No.{idx} frame start!')
            tracking_time_start = time.time()
            # Draw box

            for j, frame in enumerate(frames):
                out = trackers[j].track_one_state(frame)
                state = [int(s) for s in out['target_bbox']]
                score = round(out['conf_score'], 2)
                cv2.rectangle(frames[0], (state[0], state[1]), (state[2] + state[0], state[3] + state[1]), tracker_colors[j], 1)
                cv2.putText(frames[0], str(cameras[j]), (state[0], state[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tracker_colors[j], 1)

            cv2.imwrite(f'{save_path}/{idx:03d}.png', frames[0])
            if write_bbox:
                csv_write(save_path, state, 'bbox')

            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

    def get_parameters(self):
        """Get parameters."""
        # param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        # params = param_module.parameters(self.parameter_name)
        # return params

        AOT_PATH = os.path.join(os.path.dirname(__file__), '..')
        epoch_num = 42000
        config = {
            'exp_name': 'default',
            'model': 'internT_msdeaotl_v2',
            'pretrain_model_path': 'result/default_InternT_MSDeAOTL_V2/YTB_DAV_VIP/ckpt/save_step_{}.pth'.format(
                epoch_num),
            'gpu_id': 0, }

        from CSWinTT_multi_state.demo.configs.ytb_vip_dav_deaot_internT import EngineConfig
        # engine_config = importlib.import_module('CSWinTT_multi_state.demo.configs.' + 'ytb_vip_dav_deaot_internT')
        # cfg = engine_config.EngineConfig(config['exp_name'], config['model'])
        cfg = EngineConfig(config['exp_name'], config['model'])
        cfg.TEST_CKPT_PATH = os.path.join(AOT_PATH, config['pretrain_model_path'])

        sam_refine = True
        sam_refine_iou = 0.1
        # return config, cfg, sam_refine, sam_refine_iou
        return cfg, config, sam_refine, sam_refine_iou

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")

