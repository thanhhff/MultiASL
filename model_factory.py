
class ModelFactory:
    @staticmethod
    def get_model(model_name, config):
        # if config.video_encoder == "resnet18":
        #     config.len_feature = 512
        # elif config.video_encoder == "resnet34":
        #     config.len_feature = 512
        # elif config.video_encoder == "mobilenetv4_conv_small.e1200_r224_in1k":
        #     config.len_feature = 1280
        # elif config.video_encoder == "vit_small_patch32_224.augreg_in21k_ft_in1k":
        #     config.len_feature = 384

        if model_name == "CNN2D":
            from models.cnn2d.cnn2d import CNN2D
            return CNN2D(config.video_encoder, config.len_feature, config.num_classes, config.num_segments)
        
        elif model_name == "CNN2D_Transformer":
            from models.cnn2d_transformer.cnn2d_transformer import CNN2D_Transformer
            return CNN2D_Transformer(config.video_encoder, config.len_feature, config.num_classes, config.num_segments, config.fusion_type)
        
        elif model_name == "CNN2D_ConViT":
            from models.cnn2d_convit.cnn2d_convit import ConViT
            return ConViT(config.video_encoder, pretrained=True, num_classes=config.num_classes)
        
        elif model_name == "CNN3D": 
            from models.cnn3d.cnn3d import CNN3D
            shortcut_type = "B"
            sample_size = 224
            sample_duration=config.num_segments
            model_depth = 18
            mode = 'feature'
            return CNN3D(config.num_classes, shortcut_type, sample_size, sample_duration, model_depth, mode)
        
        elif model_name == "Query2Label":
            from models.query2label.query2label import Query2Label
            return Query2Label(config.video_encoder, config.num_classes)

        else:
            raise NotImplementedError("No such model")