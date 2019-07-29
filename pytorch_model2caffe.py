from lib.utils.pytorch2caffe import *
import os



model_path = "/data-private/nas/pspace/tiPytorchFile/mobilenetv1_voc-lite-0.5-07032330-201907041803-201907041803/yolo_v3_mobilenet_v1_lite_050_voc_epoch_1.pth"
model = torch.load(model_path)
model = model.cpu()
input_shape = [3, 512, 512]
temp = model_path.split("/")
epoch = temp[-1].split("_")[-1].replace(".pth")
save_root = os.path.join("/".join(model_path.split("/")[:-1]), "caffe_model_save"+epoch)
convert = Pytorch2Caffe(model, save_root, "ssd_save", input_shape=input_shape)
convert.start()