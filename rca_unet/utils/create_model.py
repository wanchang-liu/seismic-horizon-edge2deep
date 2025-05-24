from models.unet.unet import UNet
from models.segnet.segnet import SegNet
from models.unet.res_unet import ResUNet
from models.unet.cbam_unet import CBAMUNet
from models.unet.unet_parts import UpsampleMethod
from models.unet.cbam_res_unet import CBAMResUNet
from models.segnet.segnet_basic import SegNetBasic
from models.segnet.bayes_segnet import BayesSegNet
from models.unet.attention_unet import AttentionUNet
from models.nested_unet.nested_unet import NestedUNet
from models.unet.res_attention_unet import ResAttentionUNet
from models.unet.cbam_attention_unet import CBAMAttentionUNet
from models.unet.cbam_res_attention_unet import CBAMResAttentionUNet


# 创建模型
def create_model(args):
    if args.model == 'SegNet':
        return SegNet(args.in_channels, args.out_channels).to(args.device)
    elif args.model == 'SegNetBasic':
        return SegNetBasic(args.in_channels, args.out_channels).to(args.device)
    elif args.model == 'BayesSegNet':
        return BayesSegNet(args.in_channels, args.out_channels).to(args.device)
    elif args.model == 'NestedUNet':
        return NestedUNet(args.in_channels, args.out_channels).to(args.device)
    elif args.model == 'UNet':
        return UNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'ResUNet':
        return ResUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'CBAMUNet':
        return CBAMUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'AttentionUNet':
        return AttentionUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'CBAMResUNet':
        return CBAMResUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'ResAttentionUNet':
        return ResAttentionUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'CBAMAttentionUNet':
        return CBAMAttentionUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
    elif args.model == 'CBAMResAttentionUNet':
        return CBAMResAttentionUNet(args.in_channels, args.out_channels, UpsampleMethod.Bilinear).to(args.device)
