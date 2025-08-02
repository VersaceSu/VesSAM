import torch
import torch.nn as nn
from functools import partial
from Models import ImageEncoderViT, MaskDecoder, PromptEncoder, Ves_SAM, TwoWayTransformer
from torch.nn import functional as F
import torchvision.transforms as transforms
from Dataloader import VesselDataset
from torch.utils.data import DataLoader


def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        encoder_adapter=args.encoder_adapter,
    )


build_sam = build_sam_vit_b

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        image_size,
        encoder_adapter,
        checkpoint,
):
    prompt_image_embedding_size = 256
    image_size = image_size
    vit_patch_size = 16
    prompt_embed_dim = 64

    # image_embedding_size = image_size // vit_patch_size
    sam = Ves_SAM(
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
            out_chans=prompt_image_embedding_size,
            adapter_train=encoder_adapter,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(prompt_image_embedding_size, prompt_image_embedding_size),
            input_image_size=(image_size, image_size),
            base_chans=32,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_image_embedding_size,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_image_embedding_size,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    # sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        try:
            if 'model' in state_dict.keys():
                print(encoder_adapter)
                sam.load_state_dict(state_dict['model'], False)
            else:
                if image_size == 1024 and encoder_adapter == True:
                    sam.load_state_dict(state_dict, False)
                else:
                    sam.load_state_dict(state_dict)
        except:
            print('*******interpolate')
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
        print(f"*******load {checkpoint}")

    return sam


def load_from(sam, state_dicts, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[
                          2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if
                               '2' in k or
                               '5' in k or
                               '7' in k or
                               '8' in k or
                               '11' in k or
                               '13' in k or
                               '15' in k or
                               '23' in k or
                               '31' in k]
        # print(sam_dict)
        for k in global_rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_dir = 'data'
    dataset_name = 'Aorta'  # Example dataset
    split = 'train'  # Use 'train', 'val', or 'test' split

    dataset = VesselDataset(dataset_name=dataset_name, data_dir=dataset_dir, split=split, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 设置参数
    # embed_dim = 64  # 嵌入维度
    # image_embedding_size = (256, 256)  # 图像嵌入的空间大小 (H, W)
    # input_image_size = 256  # 输入图像的大小 (H, W)
    # mask_in_chans = 32  # 输入掩膜的通道数
    # sam_checkpoint = None

    # model = (_build_sam(
    #     encoder_embed_dim=768,
    #     encoder_depth=12,
    #     encoder_num_heads=12,
    #     encoder_global_attn_indexes=[2, 5, 8, 11],
    #     image_size=input_image_size,
    #     checkpoint=sam_checkpoint,
    #     encoder_adapter=encoder_adapter,
    # ))

    # ********************Image_Encoder***************
    image_size = 256
    vit_patch_size = 16
    encoder_embed_dim = 64
    in_chans = 3

    encoder_depth = 12
    qkv_bias = True
    norm_layer = partial(torch.nn.LayerNorm, eps=1e-6)
    encoder_num_heads = 8
    use_rel_pos = True
    window_size = 14
    encoder_global_attn_indexes = [2, 5, 8, 11]
    mlp_ratio = 4

    encoder_out_dim = 256
    adapter_train = False

    # *****************PromptEncoder*******************
    prompt_embed_dim = 64
    base_chans = 32
    prompt_activation = nn.GELU
    prompt_num_heads = 8
    prompt_num_layers = 2
    prompt_graph_layers = 2
    point_emb_chans = 256

    # *****************MaskDecoder*********************
    num_multimask_outputs = 3
    mask_transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=encoder_out_dim,
        mlp_dim=2048,
        num_heads=8,
    )
    mask_transformer_dim = encoder_out_dim
    iou_head_depth = 3
    iou_head_hidden_dim = 256

    model = Ves_SAM(
        image_encoder=ImageEncoderViT(
            img_size=image_size,
            patch_size=vit_patch_size,
            embed_dim=encoder_embed_dim,
            in_chans=in_chans,

            depth=encoder_depth,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            num_heads=encoder_num_heads,
            use_rel_pos=use_rel_pos,
            window_size=window_size,
            global_attn_indexes=encoder_global_attn_indexes,
            mlp_ratio=mlp_ratio,

            out_chans=encoder_out_dim,
            adapter_train=adapter_train,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            base_chans=base_chans,
            img_size=image_size,
            activation=prompt_activation,
            num_heads=prompt_num_heads,
            num_layers=prompt_num_layers,
            graph_layers=prompt_graph_layers,
            point_emb_chans=point_emb_chans
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_multimask_outputs,
            transformer=mask_transformer,
            transformer_dim=mask_transformer_dim,
            iou_head_depth=iou_head_depth,
            iou_head_hidden_dim=iou_head_hidden_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}")

            image = batch['image']  # Shape: [batch_size, channels, height, width]
            mask = batch['mask']  # Shape: [batch_size, 1, height, width]
            skeleton = batch['skeleton']  # Shape: [batch_size, 1, height, width]
            branch_points = batch['branch_points']  # Shape: [batch_size, max_branch_points, 2]
            mid_points = batch['mid_points']  # Shape: [batch_size, max_mid_points, 2]

            outputs = model(batch, multimask_output=True)

            # 打印输出结果
            print(
                f"masks: {outputs['masks'].shape}")  # 应该是 [batch_size, num_points*2, embed_dim][4, 40, 64]
            print(
                f"iou_predictions: {outputs['iou_predictions'].shape}")  # 应该是 [batch_size, embed_dim, 64, 64][4, 128, 64, 64]

            break
