Flat minima in the loss landscape are widely believed to promote generalization and robustness, as flatter solutions are less sensitive to parameter perturbations~\citep{FlatMinima1997,wu2017understandinggeneralizationdeeplearning,chaudhariEntropySGDBiasingGradient2019,NEURIPS2020_518a38cc,NEURIPS2021_995f5e03,JMLRAnEmpiricalInvestigatio,yangRevisitingFlatnessAwareOptimization2025}.
Sharpness-aware minimization (SAM)~\cite{foretSharpnessAwareMinimizationEfficiently2021} and related methods~\citep{pmlr-v151-bisla22a,zhangGradientNormAware2023,NEURIPS2024_C-Flat} operationalize this principle by seeking parameters with low loss  under small perturbations. 
% Sharpness-aware optimizers such as SAM~\citep{foretSharpnessAwareMinimizationEfficiently2021}, RWP~\citep{pmlr-v151-bisla22a}, GAM~\citep{zhangGradientNormAware2023} and C-FLAT~\cite{NEURIPS2024_C-Flat} instantiate this idea by explicitly penalizing increases of the loss under small parameter perturbations. 
While these methods have proven effective in full fine-tuning, their behavior in the restricted optimization landscape of PECL remains under-explored. 
% LoRA-SAM~\citep{NEURIPS2024_4eb2c0ad} provides evidence that limiting sharpness-aware updates to the LoRA coordinates already yields substantial implicit regularization, whereas Flat-LoRA~\citep{li2025flatloralowrankadaptationflat} argues that flatness restricted to the LoRA subspace may be insufficient and therefore advocates perturbing all model parameters. In the PECL regime that we consider, however, the core efficiency principle is to strictly freeze the pre-trained backbone and confine optimization to the adapter subspace $\mathcal W_\Delta$. This tension raises a central question:
% While effective in full fine-tuning, their application in PECL is not straightforward. 
Recent work presents conflicting views: LoRA-SAM~\citep{NEURIPS2024_4eb2c0ad} suggests that optimizing flatness within the LoRA subspace alone provides effective regularization, while Flat-LoRA~\citep{li2025flatloralowrankadaptationflat} argues that achieving sufficient flatness requires perturbing all model parameters, which implicitly includes the frozen backbone. The latter requirement, however, contradicts the fundamental PECL constraint of a fixed backbone, thus motivating our central question:


% \emph{In PECL, is it necessary to pursue global flatness by perturbing all model parameters, or is controlling flatness solely within the shared adapter subspace sufficient to govern stability and plasticity?}

\emph{In PECL with a frozen backbone, is controlling flatness within the adapter subspace sufficient to govern stability and plasticity, or is flatness control induced by full-parameter perturbation necessary?}

To address this question, we develop a sequential hierarchical PAC Bayes analysis tailored to PECL with a frozen backbone. Unlike classical hierarchical bounds that assume fixed hyper distribution, our formulation allows the task level hyperposterior to evolve with the task index and introduces an explicit drift penalty that quantifies cross task interference. We then specialize the framework to LoRA based PECL and show that, under the frozen backbone constraint, both the KL complexity and the sharpness dependent empirical term reduce exactly to quantities defined on the LoRA update subspace.


Empirically, our results support a single message: in frozen-backbone PECL, the geometry that matters is the trainable adapter subspace. Adapter-only sharpness-aware optimization consistently improves continual performance and robustness, while full-space perturbations that include the frozen backbone do not yield systematic benefits and often harm stability. 





\begin{figure}[tbp]
\centering
\begin{minipage}{1\linewidth}
\centering
\includegraphics[width=1\linewidth]{figures_TRML/2D_lossLandscape_task0_lora_opt_sam_vs_sgd_3d_compare_withxyz.pdf}
\caption{Visualization of the loss landscape for a ViT backbone trained with LoRA adapters, comparing SGD and SAM optimizers. Applying SAM to adapter parameters yields a flatter local landscape around the final solution.}
% The SAM optimizer not only discovers a wider and flatter basin but also achieves a lower loss value
\label{fig:3d_loss_landscape}
\end{minipage}
\end{figure}