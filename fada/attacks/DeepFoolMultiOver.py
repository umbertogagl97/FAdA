from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from fada.utils import *

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, is_probability

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class DeepFool_mod(EvasionAttack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).
    | Paper link: https://arxiv.org/abs/1511.04599
    """

    attack_params = EvasionAttack.attack_params + [
        "max_iter",
        "epsilon",
        "nb_grads",
        "class_target",
        "confidence",
        "random_mask",
        "max_over",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        max_iter: int = 100,
        epsilon: float = 1e-6,
        nb_grads: int = 10,
        class_target = 0,
        confidence = 0.7,
        random_mask = False,
        max_over = 1,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a DeepFool attack instance.
        :param classifier: A trained classifier.
        :param max_iter: The maximum number of iterations.
        :param epsilon: Overshoot parameter.
        :param nb_grads: The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the
                         most likely classes are considered, speeding up the computation.
        :param class_target: classe da far predire.
        :param confidence: probabilità minima con la quale predire class_target 
        :param random_mask:                         
        :param max_over: numero massimo di volte che può applicare il parametro overshoot
        :param batch_size: Batch size
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.nb_grads = nb_grads
        self.class_target=class_target
        self.confidence=confidence
        self.random_mask=random_mask
        self.batch_size = batch_size
        self.max_over = max_over
        self.verbose = verbose
        self._check_params()
        if self.estimator.clip_values is None:
            logger.warning(
                "The `clip_values` attribute of the estimator is `None`, therefore this instance of DeepFool will by "
                "default generate adversarial perturbations scaled for input values in the range [0, 1] but not clip "
                "the adversarial example."
            )

    def generate(self, mask_mod, x: np.ndarray, y: Optional[np.ndarray] = None, enh=False, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.
        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        x_adv = x.astype(ART_NUMPY_DTYPE)

        size_init=np.array(x.shape[2:4])
        transf_orig=transforms.Resize(size=(size_init[0],size_init[1]),interpolation=InterpolationMode.NEAREST)

        #preds = self.estimator.predict(x, batch_size=self.batch_size)

        class_pred,prob_preds,preds=test_average(self.estimator,torch.Tensor(x_adv),transf_orig)
        preds=np.array(preds).reshape(1,2)

        if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if is_probability(preds[0]):
            logger.warning(
                "It seems that the attacked model is predicting probabilities. DeepFool expects logits as model output "
                "to achieve its full attack strength."
            )

        # Determine the class labels for which to compute the gradients
        use_grads_subset = self.nb_grads < self.estimator.nb_classes
        if use_grads_subset:
            # TODO compute set of unique labels per batch
            grad_labels = np.argsort(-preds, axis=1)[:, : self.nb_grads]
            labels_set = np.unique(grad_labels)
        else:
            labels_set = np.arange(self.estimator.nb_classes)
        sorter = np.arange(len(labels_set))

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        if ((class_pred==np.argmax(y, axis=1)) and (class_pred!= self.class_target)):
            active=True
        elif ((class_pred!=np.argmax(y, axis=1)) and (class_pred== self.class_target) and (np.max(prob_preds)<self.confidence)): 
            active=True
        else:  active=False

        if active: 
            x_adv=transf_resize(torch.Tensor(x_adv))
            #mask_mod=compute_mask(x_adv)
            x_adv=np.array(x_adv)

            x_init=transf_resize(torch.Tensor(x.astype(ART_NUMPY_DTYPE)))
            x_init=np.array(x_init)

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="DeepFool", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2].copy()

            # Get predictions and gradients for batch
            #f_batch = preds[batch_index_1:batch_index_2] #predizioni
            #fk_hat = np.argmax(f_batch, axis=1) #classe predetta
            
            batch_grd=np.array(trans_norm(torch.Tensor(batch)))
            if use_grads_subset:
                # Compute gradients only for top predicted classes
                grd = np.array([self.estimator.class_gradient(batch_grd, label=_) for _ in labels_set])
                grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
            else:
                # Compute gradients for all classes
                grd = self.estimator.class_gradient(batch_grd)

            # Get current predictions

            #active_indices = np.arange(len(batch))
            current_step = 0

            while active==True and current_step < self.max_iter:
                # Compute difference in predictions and gradients only for selected top predictions
                labels_indices = sorter[np.searchsorted(labels_set, class_pred, sorter=sorter)]
                #rand_mask=rand_roi(mask_mod)
                if self.random_mask: m=rand_roi(mask_mod)
                else: m=mask_mod.copy()
                grad_diff = (grd - grd[np.arange(len(grd)), labels_indices][:, None])*m
                f_diff = preds[:,labels_set] - preds[np.arange(len(preds)), labels_indices][:, None]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2) + tol
                value = np.abs(f_diff) / norm
                value[np.arange(len(value)), labels_indices] = np.inf
                l_var = np.argmin(value, axis=1)
                absolute1 = abs(f_diff[np.arange(len(f_diff)), l_var])
                draddiff = grad_diff[np.arange(len(grad_diff)), l_var].reshape(len(grad_diff), -1)
                pow1 = (
                    pow(
                        np.linalg.norm(draddiff, axis=1),
                        2,
                    )
                    + tol
                )
                r_var = absolute1 / pow1
                r_var = r_var.reshape((-1,) + (1,) * (len(x.shape) - 1))
                r_var = r_var * grad_diff[np.arange(len(grad_diff)), l_var]
                #print("pert")
                #plt.imshow(r_var[0].transpose(1,2,0)*255)
                #plt.show()
                r, g, b = r_var[0,0,:,:],r_var[0,1,:,:],r_var[0,2,:,:]
                r_var = 0.2989 * r + 0.5870 * g + 0.1140 * b

                # Add perturbation and clip result
                if self.estimator.clip_values is not None:
                    batch = np.clip(
                        batch
                        + r_var * (self.estimator.clip_values[1] - self.estimator.clip_values[0]),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1]
                    )
                else:
                    batch += r_var
                
                batch=batch.astype(np.single)
                if enh: batch=enhanc(batch,mask_mod)
                # Recompute prediction for new x
                class_pred_i,prob_preds,preds=test_average(self.estimator,torch.Tensor(batch),transf_orig)
                preds=np.array(preds).reshape(1,2)
                #f_batch = self.estimator.predict(batch)
                #fk_i_hat = np.argmax(f_batch, axis=1)
                
                batch_grd=np.array(trans_norm(torch.Tensor(batch)))
                # Recompute gradients for new x
                if use_grads_subset:
                    # Compute gradients only for (originally) top predicted classes
                    grd = np.array([self.estimator.class_gradient(batch_grd, label=_) for _ in labels_set])
                    grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
                else:
                    # Compute gradients for all classes
                    grd = self.estimator.class_gradient(batch_grd)

                # Stop if misclassification has been achieved
                #active_indices = np.where(class_pred_i == class_pred)[0]
                if ((class_pred_i==np.argmax(y, axis=1)) and (class_pred_i!= self.class_target)):
                  active=True
                elif ((class_pred_i!=np.argmax(y, axis=1)) and (class_pred_i== self.class_target) and (np.max(prob_preds)<self.confidence)): 
                  active=True
                else:  active=False

                current_step += 1
            
            ov_it=0
            while active and ov_it<self.max_over:
              x_adv2 = (1 + self.epsilon) * (batch - x_init)            
              batch = x_init + x_adv2
              for l in range(3):
                batch[0,l,:,:]=np.where(mask_mod==0,1,batch[0,l,:,:])
              if self.estimator.clip_values is not None:
                  np.clip(
                      batch,
                      self.estimator.clip_values[0],
                      self.estimator.clip_values[1],
                      out=batch,
                  )   
              if enh: batch=enhanc(batch,mask_mod)
              class_pred,prob_preds,preds=test_average(self.estimator,torch.Tensor(batch),transf_orig)
              if ((class_pred==np.argmax(y, axis=1)) and (class_pred!= self.class_target)):
                  active=True
              elif ((class_pred!=np.argmax(y, axis=1)) and (class_pred== self.class_target) and (np.max(prob_preds)<self.confidence)): 
                  active=True
              else:  active=False    
              ov_it+=1
              
            x_adv=batch
        '''      
        logger.info(
            "Success rate of DeepFool attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, batch_size=self.batch_size),
        )
        '''
        return x_adv

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.nb_grads, (int, np.int)) or self.nb_grads <= 0:
            raise ValueError("The number of class gradients to compute must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if isinstance(self.class_target,int):
            if self.class_target!=0 and self.class_target!=1:
              raise ValueError("Attacco destinato alle impronte (0,1)")
        else: 
          raise ValueError("Class_target deve essere intero")   

        if isinstance(self.confidence,float):
            if self.confidence <=0 or self.confidence >1:
              raise ValueError("confidence deve essere compreso tra 0 e 1")
        else: 
          raise ValueError("confidence deve essere float")

        if isinstance(self.max_over,int):
            if self.max_over <=0:
              raise ValueError("max_over deve essere positivo")
        else: 
          raise ValueError("confidence deve essere int")
