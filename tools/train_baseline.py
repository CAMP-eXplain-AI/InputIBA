from captum.attr import GuidedBackprop
from captum.attr import DeepLiftShap
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward

class MethodWrapper():
    def __init__(self, model, method, saliency_layer=None):
        self.model = model.eval()
        self.method = method
        if method == "extremal perturbation":
            def make_attribution(model):
                def attribution_func(input, target):
                    saliency_map, _ = extremal_perturbation(model, input, target)
                    return saliency_map
                return attribution_func
            self.attributer = make_attribution(self.model)

        elif method == "grad cam":
            assert saliency_layer,  "Please give a saliency layer!"
            def make_attribution(model, saliency_layer):
                def attribution_func(input, target):
                    saliency_map = grad_cam(model, input, target, saliency_layer=saliency_layer)
                    return saliency_map
                return attribution_func
            self.attributer = make_attribution(self.model, saliency_layer)

        elif method == "deep shap":
            self.attributer = DeepLiftShap(self.model)

        elif method == "guided backpropagation":
            self.attributer = GuidedBackprop(self.model)

        else:
            raise NotImplementedError

    def get_attribution(self, img, target):
        if self.method == "grad cam" or self.method == "extremal perturbation":
            saliency_map = self.attributer(img, target)
        else:
            saliency_map = self.attributer.attribute(img, target)
        return saliency_map
