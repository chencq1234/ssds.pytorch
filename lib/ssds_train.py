from __future__ import print_function
import numpy as np
import os
import sys
import cv2
import random
import pickle
from lib.param_summary import torch_summarize_df
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init
from lib.calpr.pr_curve import process_pr_curve, PRargs
from tensorboardX import SummaryWriter

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from lib.dataset.voc import VOC_CLASSES
import logging
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # "0,1,2,3,4,5,6,7"

# ##############################  plot path and num  #################################################################
def creat_log(cfg, phase="train"):
    now = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
    if phase == "train":
        cfg['EXP_DIR'] = "".join([cfg['EXP_DIR'], "-", now])
        cfg['LOG_DIR'] = "".join([cfg['LOG_DIR'], "-", now])

    path = os.path.join(cfg['LOG_DIR'], ''.join([cfg["CHECKPOINTS_PREFIX"], now, ".txt"]))
    print("log path is:\n %s" % path)
    if not os.path.exists(cfg['LOG_DIR']):
        os.makedirs(cfg['LOG_DIR'])
        print(cfg['LOG_DIR'] + ' --creat dir successfully')
    else:
        print(cfg['LOG_DIR'] + ' --dir already exist')

    logging.basicConfig(level=logging.INFO,
                        filename=path,
                        filemode='w',
                        # format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    return now
# ##############################  end  #############################################################################


class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self, phase="train"):
        self.cfg = cfg
        creat_log(self.cfg, phase=phase)
        for k, v in cfg.items():
            print(k, ": ", v)
            log_str = '\rEpoch {k}: {v}'.format(k=k, v=v)
            logging.info(log_str)
            # Load data
        print('===> Loading data')
        logging.info('===> Loading data')
        self.train_loader = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        self.test_loader = load_data(cfg.DATASET, 'test') if 'test' in cfg.PHASE else None
        self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        # Build model
        print('===> Building model')
        logging.info('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.detector = Detect(cfg.POST_PROCESS, self.priors)
        os.makedirs(self.cfg['EXP_DIR'], exist_ok=True)

        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        print('Model architectures:\n{}\n'.format(self.model))
        logging.info('Model architectures:\n{}\n'.format(self.model))

        from lib.utils.torchsummary import summary
        summary_text = summary(self.model.cuda(), (3,  cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]))
        logging.info('\n'.join(summary_text))
        # num_params = 0
        # for name, param in self.model.named_parameters():
        #     num_params += param.numel()
        #     # print(name, param.size(), param.numel())
        #     print("%40s %20s  %20s" % (name, num_params, param.numel()))
        # print(num_params/1e4)
        # df = torch_summarize_df(input_size=(3, 512, 512), model=self.model)
        # # df['name'], list(df['class_name']), df['input_shape'], df["output_shape"], list(df['nb_params'])
        # print(df)
        # for name, param in self.model.named_parameters():
        #     print(name, param.size())
        # from thop import profile
        #
        # flops, params = profile(self.model, input_size=(1, 3, 512, 128))
        # count = 0
        # for p in self.model.parameters():
        #     count += p.data.nelement()
        # self.multi_gpu = True
        self.multi_gpu = False
        if self.use_gpu:
            print('Utilize GPUs for computation')
            logging.info('Utilize GPUs for computation')
            # print('Number of GPU available', torch.cuda.device_count())
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
            # os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"  # "0,1,2,3,4,5,6,7"
            if torch.cuda.device_count() > 1 and self.multi_gpu:
                self.model = torch.nn.DataParallel(self.model.cuda())
                cudnn.benchmark = True
                # self.model = torch.nn.DataParallel(self.model).module

        # Print the model architecture and parameters

        # print('Parameters and size:')
        # for name, param in self.model.named_parameters():
        #     print('{}: {}'.format(name, list(param.size())))

        # print trainable scope
        print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        logging.info('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param = self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)
        self.optimizer = self.configure_optimizer(trainable_param, cfg.TRAIN.OPTIMIZER)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = MultiBoxLoss(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(logdir=cfg.LOG_DIR)
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX

    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filename)
        # torch.save(self.model, filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))
        logging.info('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        logging.info(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        # change the name of the weights which exists in other model
        # change_dict = {
        #         'conv1.weight':'base.0.weight',
        #         'bn1.running_mean':'base.1.running_mean',
        #         'bn1.running_var':'base.1.running_var',
        #         'bn1.bias':'base.1.bias',
        #         'bn1.weight':'base.1.weight',
        #         }
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k == k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)
        # change_dict = {'layer1.{:d}.'.format(i):'base.{:d}.'.format(i+4) for i in range(20)}
        # change_dict.update({'layer2.{:d}.'.format(i):'base.{:d}.'.format(i+7) for i in range(20)})
        # change_dict.update({'layer3.{:d}.'.format(i):'base.{:d}.'.format(i+11) for i in range(30)})
        # for k, v in list(checkpoint.items()):
        #     for _k, _v in list(change_dict.items()):
        #         if _k in k:
        #             new_key = k.replace(_k, _v)
        #             checkpoint[new_key] = checkpoint.pop(k)

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
            # for k, v in checkpoint._modules.items():
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}
        # print("=> Resume weigths:")
        # print([k for k, v in list(pretrained_dict.items())])

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)
            logging.info("=> UNResume weigths:")
            logging.info(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            logging.info('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        model = self.model.module if self.multi_gpu else self.model
        for param in model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(model, module):
                # print(getattr(self.model, module))
                for param in getattr(model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(model, module).parameters())

        return trainable_param

    def train_model(self):
        self.export_graph()
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        aps_list, map_list = [], []
        apvoc_list = []
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)
            if 'train' in cfg.PHASE:
                self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
            if 'test' in cfg.PHASE:
                # if epoch % 20 != 10:
                #     continue
                aps, map = self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, epoch, need_pr=True)
                if map is not None:
                    aps_list.append(aps)
                    map_list.append(map)
                    max_idx1 = np.argmax(np.array(aps_list)[:, 0])
                    max_map_idx = np.argmax(np.array(map_list))
                    print("ap1 max: %f , epoch is: %d" % (aps_list[max_idx1][0], start_epoch+1+max_idx1))
                    print("map max: %f , epoch is: %d" % (map_list[max_map_idx], start_epoch+1+max_map_idx))
                    logging.info("ap1 max: %f , epoch is: %d" % (aps_list[max_idx1][0], start_epoch+1+max_idx1))
                    logging.info("map max: %f , epoch is: %d" % (map_list[max_map_idx], start_epoch+1+max_map_idx))

                pr_path = os.path.join(self.output_dir, str(epoch))
                pr_intput = PRargs(detFolder=pr_path)
                res_dict = process_pr_curve(pr_intput)
                apvoc_list.append(res_dict['pbox'])
                print("-----------------------------------------------------------------")
                print('cls {} ap: {}'.format('pbox', res_dict['pbox']))
                print("-----------------------------------------------------------------")
                logging.info("-----------------------------------------------------------------")
                logging.info('cls {} ap: {}'.format('pbox', res_dict['pbox']))
                logging.info("-----------------------------------------------------------------")
                max_vocap_idx = np.argmax(apvoc_list)
                print("ap1 voc max: %f , epoch is: %d" % (apvoc_list[max_vocap_idx], start_epoch+1+max_vocap_idx))
                logging.info("ap1 voc max: %f , epoch is: %d" % (apvoc_list[max_vocap_idx], start_epoch+1+max_vocap_idx))

            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0 or start_epoch+1+max_vocap_idx == epoch:
                self.save_checkpoints(epoch)

    def test_model(self, need_pr=True):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
                    self.resume_checkpoint(resume_checkpoint)
                    if 'eval' in cfg.PHASE:
                        self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
                    if 'test' in cfg.PHASE:
                        self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu, epoch, need_pr)
                    if 'visualize' in cfg.PHASE:
                        self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            epoch = self.checkpoint.split('_')[-1].replace(".pth", "")
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
            if 'test' in cfg.PHASE:
                self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu, epoch, need_pr)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu):
        model.train()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        for iteration in iter(range((epoch_size))):
            # if iteration > 8: break# print("iteration:", iteration)
            images, targets = next(batch_iterator)
            if use_gpu:
                images = images.cuda()
                with torch.no_grad():
                    targets = [anno.cuda() for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            _t.tic()
            # forward
            # if (3,512,512) != (images.shape[1],images.shape[2],images.shape[3]):
            #     print(111, )
            out = model(images, phase='train')

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)

            # some bugs in coco train2017. maybe the annonation bug.
            # if loss_l.data[0] == float("Inf"):
            if loss_l.data.item() == float("Inf"):
                continue

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            time = _t.toc()
            loc_loss += loss_l.data.item()
            conf_loss += loss_c.data.item()

            # log per iter
            log = '\r==>Train: || epoch: {epoch:4d} || {iters:4d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data.item(), cls_loss=loss_c.data.item(), epoch=epoch)
            print(log)
            logging.info(log)
            # sys.stdout.write(log)
            # sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        logging.info(log)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)


    def eval_epoch(self, model, data_loader, detector, criterion, writer, epoch, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        label = [list() for _ in range(model.num_classes)]
        gt_label = [list() for _ in range(model.num_classes)]
        score = [list() for _ in range(model.num_classes)]
        size = [list() for _ in range(model.num_classes)]
        npos = [0] * model.num_classes

        for iteration in iter(range((epoch_size))):
        # for iteration in iter(range((10))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            _t.tic()
            # forward
            out = model(images, phase='train')

            # loss
            loss_l, loss_c = criterion(out, targets)

            out = (out[0], model.softmax(out[1].view(-1, model.num_classes)))

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # evals
            label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
            size = cal_size(detections, targets, size)
            loc_loss += loss_l.data.item()
            conf_loss += loss_c.data.item()

            # log per iter
            log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data.item(), cls_loss=loss_c.data.item())
            logging.info(log)
            sys.stdout.write(log)
            sys.stdout.flush()

        # eval mAP
        prec, rec, ap = cal_pr(label, score, npos)

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        logging.info(log)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Eval/mAP', ap, epoch)
        viz_pr_curve(writer, prec, rec, epoch)
        viz_archor_strategy(writer, size, gt_label, epoch)

    # TODO: HOW TO MAKE THE DATALOADER WITHOUT SHUFFLE
    # def test_epoch(self, model, data_loader, detector, output_dir, use_gpu):
    #     # sys.stdout.write('\r===> Eval mode\n')

    #     model.eval()

    #     num_images = len(data_loader.dataset)
    #     num_classes = detector.num_classes
    #     batch_size = data_loader.batch_size
    #     all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    #     empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

    #     epoch_size = len(data_loader)
    #     batch_iterator = iter(data_loader)

    #     _t = Timer()

    #     for iteration in iter(range((epoch_size))):
    #         images, targets = next(batch_iterator)
    #         targets = [[anno[0][1], anno[0][0], anno[0][1], anno[0][0]] for anno in targets] # contains the image size
    #         if use_gpu:
    #             images = Variable(images.cuda())
    #         else:
    #             images = Variable(images)

    #         _t.tic()
    #         # forward
    #         out = model(images, is_train=False)

    #         # detect
    #         detections = detector.forward(out)

    #         time = _t.toc()

    #         # TODO: make it smart:
    #         for i, (dets, scale) in enumerate(zip(detections, targets)):
    #             for j in range(1, num_classes):
    #                 cls_dets = list()
    #                 for det in dets[j]:
    #                     if det[0] > 0:
    #                         d = det.cpu().numpy()
    #                         score, box = d[0], d[1:]
    #                         box *= scale
    #                         box = np.append(box, score)
    #                         cls_dets.append(box)
    #                 if len(cls_dets) == 0:
    #                     cls_dets = empty_array
    #                 all_boxes[j][iteration*batch_size+i] = np.array(cls_dets)

    #         # log per iter
    #         log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
    #                 prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
    #                 time=time)
    #         sys.stdout.write(log)
    #         sys.stdout.flush()

    #     # write result to pkl
    #     with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
    #         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    #     print('Evaluating detections')
    #     data_loader.dataset.evaluate_detections(all_boxes, output_dir)

    def inference(self, model, images_dir, detector, output_dir, use_gpu, epoch, need_pr=None):
        model.eval()
        images_list = os.listdir(images_dir)
        num_images = len(images_list)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

        _t = Timer()
        if need_pr:
            pr_path = os.path.join(output_dir, str(epoch))
            os.makedirs(pr_path, exist_ok=True)

            print("test epoch %s pr result path: %s" %(epoch, pr_path))
            logging.info("test epoch %s pr result path: %s" %(epoch, pr_path))
        for i, image in enumerate(images_list):
            img = cv2.imread(os.path.join(images_dir, image))
            img_key = image.replace(".jpg", "")
            scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            # if use_gpu:
            #     images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
            # else:
            # images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

            _t.tic()
            # forward
            out = model(img, phase='eval')

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # TODO: make it smart:
            pr_arr = []
            for j in range(1, num_classes):
                cls_dets = list()
                cls_name = VOC_CLASSES[j]
                for det in detections[0][j]:
                    if det[0] > 0:
                        d = det.cpu().numpy()
                        score, box = d[0], d[1:]
                        box *= scale
                        box = np.append(box, score)
                        cls_dets.append(box)
                        if need_pr:
                            det_res4pr = [cls_name] + [str(i) for i in d]
                            pr_arr.append(' '.join(det_res4pr))

                if len(cls_dets) == 0:
                    cls_dets = empty_array
                all_boxes[j][i] = np.array(cls_dets)
            if need_pr:
                pr_res_str = '\n'.join(pr_arr)
                with open(os.path.join(pr_path, img_key + ".txt"), "w") as f:
                    f.write(pr_res_str)
            # log per iter
            log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
                prograss='#' * int(round(10 * i / num_images)) + '-' * int(round(10 * (1 - i / num_images))), iters=i,
                epoch_size=num_images,
                time=time)
            logging.info(log)
            sys.stdout.write(log)
            sys.stdout.flush()

    def test_epoch(self, model, data_loader, detector, output_dir, use_gpu, epoch, need_pr=None):
        model.eval()
        dataset = data_loader.dataset
        num_images = len(dataset)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

        _t = Timer()
        if need_pr:
            pr_path = os.path.join(output_dir, str(epoch))
            os.makedirs(pr_path, exist_ok=True)
            print("test epoch %s pr result path:\n %s" % (epoch, pr_path))
            logging.info("test epoch %s pr result path: %s" % (epoch, pr_path))
        for i in iter(range(num_images)):
            img = dataset.pull_image(i)
            if dataset.name == 'COCO':
                img_key = dataset.image_indexes[i]
            else:
                img_key = dataset.ids[i][-1]
            scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            if use_gpu:
                with torch.no_grad():
                    images = dataset.preproc(img)[0].unsqueeze(0).cuda()
            else:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

            _t.tic()
            # forward
            out = model(images, phase='eval')

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # TODO: make it smart:
            pr_arr = []
            for j in range(1, num_classes):
                cls_dets = list()
                cls_name = VOC_CLASSES[j]
                for det in detections[0][j]:
                    if det[0] > 0:
                        d = det.cpu().numpy()
                        score, box = d[0], d[1:]
                        box *= scale
                        box = np.append(box, score)
                        cls_dets.append(box)
                        if need_pr:
                            det_res4pr = [cls_name] + [str(i) for i in d]
                            pr_arr.append(' '.join(det_res4pr))


                if len(cls_dets) == 0:
                    cls_dets = empty_array
                all_boxes[j][i] = np.array(cls_dets)
            if need_pr:
                pr_res_str = '\n'.join(pr_arr)
                with open(os.path.join(pr_path, img_key+".txt"), "w") as f:
                    f.write(pr_res_str)
            # log per iter
            log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]\r'.format(
                    prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
                    time=time)
            # logging.info(log)
            sys.stdout.write(log)
            sys.stdout.flush()
        # write result to pkl
        with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
        print('Evaluating detections')
        logging.info('Evaluating detections')
        aps, map = [], None
        aps, map = data_loader.dataset.evaluate_detections(all_boxes, output_dir)

        return aps, map

    def visualize_epoch(self, model, data_loader, priorbox, writer, epoch, use_gpu):
        model.eval()

        img_index = random.randint(0, len(data_loader.dataset)-1)

        # get img
        image = data_loader.dataset.pull_image(img_index)
        anno = data_loader.dataset.pull_anno(img_index)

        # visualize archor box
        viz_prior_box(writer, priorbox, image, epoch)

        # get preproc
        preproc = data_loader.dataset.preproc
        preproc.add_writer(writer, epoch)
        # preproc.p = 0.6

        # preproc image & visualize preprocess prograss
        images = Variable(preproc(image, anno)[0].unsqueeze(0), volatile=True)
        if use_gpu:
            images = images.cuda()

        # visualize feature map in base and extras
        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model.module(images, 'feature'), module_name='feature_extractors', epoch=epoch)

        model.train()
        images.requires_grad = True
        images.volatile=False
        base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)

        # TODO: add more...


    def configure_optimizer(self, trainable_param, cfg):
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS//2)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


    def export_graph(self):
        self.model.train(False)
        # if not self.multi_gpu:  # TODO bugs not support multi gpu
        if 0:  # TODO bugs not support multi gpu
        # torch.save(self.model, os.path.join(self.cfg["EXP_DIR"], "model.pth"))
            dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
            # Export the model
            save_onnx_path = os.path.join(self.cfg["EXP_DIR"], "graph.onnx")
            torch_out = torch.onnx._export(self.model,             # model being run
                                           dummy_input,            # model input (or a tuple for multiple inputs)
                                           save_onnx_path,         # where to save the model (can be a file or file-like object)
                                           export_params=True)     # store the trained parameter weights inside the model file
            print("----------------------------------------------------------------")
            print("save model to onnx:%s" %save_onnx_path)
            logging.info("----------------------------------------------------------------")
            logging.info("save model to onnx:%s" %save_onnx_path)

        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))


def train_model():
    s = Solver(phase="train")
    # s.export_graph()
    s.train_model()
    return True

def test_model():
    s = Solver(phase="test")
    s.test_model()
    return True
