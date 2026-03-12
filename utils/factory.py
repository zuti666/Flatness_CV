def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models_CL.simplecil import Learner
    elif name == "ogd":
        from models_CL.OGD import Learner
    elif name == "ogd_collector":
        from models_CL.OGD_collector import Learner
    elif name == "ewc":
        from models_CL.EWC import Learner
    elif name == "ewcon":
        from models_CL.EWCon import Learner
    elif name in {"ewcon_rwp_gaussian", "ewcon_rwp_gauss"}:
        from models_CL.EWCon_RWP_Gaussian import Learner
    elif name in {"ewcon_rwp_fisher", "ewcon_fisher"}:
        from models_CL.EWCon_RWP_Fisher import Learner
    elif name in {"fo_so", "fo_so_noise", "ogd_noiseinject", "ogd_noise_inject"}:
        from models_CL.OGD_noiseInject import Learner
    elif name in {"ogd_noiseinject_gaussian", "ogd_noise_inject_gaussian", "fo_so_noise_gaussian"}:
        from models_CL.OGD_noiseInject_gaussian import Learner
    elif name in {"ogd_rwp_gaussian", "ogd_rwp", "fo_so_rwp_gaussian"}:
        from models_CL.OGD_RWP_Gaussian import Learner
    elif name in {"rwp_gaussian", "fo_so_rwp_gaussian_noogd"}:
        from models_CL.RWP_Gaussian import Learner
    elif name in {"ogd_rwp_fisher", "fo_so_rwp_fisher", "ogd_rwp_fisher_noise"}:
        from models_CL.OGD_RWP_Fisher import Learner
    elif name in {"rwp_fisher", "fo_so_rwp_fisher_noogd"}:
        from models_CL.RWP_Fisher import Learner
    elif name in {"ogd_rwp_fisher_2", "fo_so_rwp_fisher_2", "ogd_rwp_fisher_v2"}:
        from models_CL.OGD_RWP_Fisher_2 import Learner
    elif name in {"ogd_fisher3", "ogd_rwp_fisher_3", "fo_so_rwp_fisher_3", "ogd_fisher_3"}:
        from models_CL.OGD_Fisher3 import Learner
    elif name in {"rwp_fisher_2", "fo_so_rwp_fisher_2_noogd", "rwp_fisher2"}:
        from models_CL.RWP_Fisher_2 import Learner
    elif name in {"fo_so_noise_optimizer", "fo_so_noise_opt", "ogd_noiseinject_optimizer", "ogd_noise_inject_optimizer"}:
        from models_CL.OGD_noiseInject_optimizer import Learner
    elif name == "si":
        from models_CL.SI import Learner
    elif name == "si_ewc":
        from models_CL.SI_EWC import Learner
    elif name == "si_ewcon":
        from models_CL.SI_EWCon import Learner
    elif name == "gpm":
        from models_CL.GPM import Learner
    elif name == "gpm_ewc":
        from models_CL.GPM_EWC import Learner
    elif name == "agem":
        from models_CL.AGEM import Learner
    elif name == "gem":
        from models_CL.GEM import Learner
    elif name in {"gem_noise", "gem-noise", "gemnoise"}:
        from models_CL.GEM_noise import Learner
    elif name in {"gem_noise_pert", "gem-noise-pert", "gemnoisepert"}:
        from models_CL.GEM_noise_pert import Learner
    elif name == "gem_ewc":
        from models_CL.GEM_EWC import Learner
    elif name == "piece":
        from models_CL.PIECE import Learner
    elif name == "fopng":
        from models_CL.FOPNG import Learner
    elif name == "flad":
        from models_CL.FLAD import Learner
    elif name in {"flad_filterogdfisher", "flad_filter_ogd_fisher", "flad-ogd-fisher"}:
        from models_CL.FLAD_filterogdfisher import Learner
    elif name in {"flad_approxminte", "flad_fd", "flad-fd"}:
        from models_CL.FLAD_approxminte import Learner
    elif name in {
        "flad_secondpreviousfishercontrol",
        "flad_secondprevious_fisher_control",
        "flad_spfc",
        "flad-secondpreviousfishercontrol",
    }:
        from models_CL.FLAD_SecondPreviousFisherControl import Learner
    elif name in {"sam_ogd", "samogd"}:
        from models_CL.SAM_OGD import Learner
    elif name in {"gam_ogd_fisher", "gam_ogd_fishermethod", "gamogdfisher"}:
        from models_CL.GAM_OGD_FisherMethod import Learner
    elif name == "geolora":
        from models.GeoLoRA import Learner
    # loraBased
    elif name == "sdlora":
        from models_LoRAbasedCL.sdlora import Learner
    elif name == "ogd_lora":
        from models_LoRAbasedCL.OGD_LoRA import Learner
    elif name == "ewc_lora":
        from models_LoRAbasedCL.EWC_LoRA_AB import Learner
    elif name == "ewc_lora_ab":
        from models_LoRAbasedCL.EWC_LoRA_AB import Learner
    elif name == "ewclora":
        from models_LoRAbasedCL.ewclora_origin import Learner
    elif name == "ewc_yaoyue_lora_liying":
        from models_LoRAbasedCL.EWC_Yaoyue_LoRA_liying import Learner
    elif name == "ewclora_youyue_github":
        from models_LoRAbasedCL.ewclora_youyue_github import Learner
    elif name == "fo_so_lora":
        from models_LoRAbasedCL.OGD_EWC_LoRA import Learner
    elif name == "si_lora":
        from models_LoRAbasedCL.SI_AB_LoRA import Learner
    elif name == "seqlora":
        from models_LoRAbasedCL.seqlora import Learner
    elif name == "inclora":
        from models_LoRAbasedCL.inclora import Learner
    elif name == "olora":
        from models_LoRAbasedCL.olora import Learner
    elif name == "inflora":
        from models_LoRAbasedCL.inflora import Learner
    elif name == "eflora":
        from models_LoRAbasedCL.eflora import Learner
    elif name == "infogam_lora":
        from models_LoRAbasedCL.infogam_lora import Learner
    elif name == "infogamlora":
        from models_LoRAbasedCL.infogam_lora import Learner
    elif name == "infobudget_gam_lora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudgetgamlora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudget_lora":
        from models_LoRAbasedCL.infobudget_gam_lora import Learner
    elif name == "infobudget_gam_lora_subnce":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_gam_lora_nce":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_subnce_lora":
        from models_LoRAbasedCL.infobudget_gam_lora_subnce import Learner
    elif name == "infobudget_gam_rankfreeze_lora":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudget_rankfreeze_lora":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudgetgamlorarankfreeze":
        from models_LoRAbasedCL.infobudget_gam_rankfreeze_lora import Learner
    elif name == "infobudget_gam_rankrecycle_lora":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infobudget_rankrecycle_lora":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infobudgetgamlorarankrecycle":
        from models_LoRAbasedCL.infobudget_gam_rankrecycle_lora import Learner
    elif name == "infoprob_gam_lora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "infoprob_lora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "infoprobgamlora":
        from models_LoRAbasedCL.infoprob_gam_lora import Learner
    elif name == "fcam_lora":
        from models_LoRAbasedCL.fcam_lora import Learner
    elif name == "fcamgamlora":
        from models_LoRAbasedCL.fcam_lora import Learner
    elif name == "geolora":
        from models.GeoLoRA import Learner
    elif name == "geolora_slowmerge":
        from models_LoRAbasedCL.GeoLoRA_SlowMerge import Learner
    elif name == "geolora_origin":
        from models_LoRAbasedCL.GeoLoRA_origin import Learner
    elif name == "geo_inc_lora":
        from models_LoRAbasedCL.geo_inc_lora import Learner
    elif name == "geoinclora":
        from models_LoRAbasedCL.geo_inc_lora import Learner
    elif name == "fcam":
        from models_LoRAbasedCL.fcam_lora import Learner


    elif name == "finetune":
        from models_CL.finetune import Learner
    elif name in {"scm_finetune", "scmfinetune", "scm"}:
        from models_CL.scm_finetune import Learner
    elif name == "LPFT":
        from models_CL.LPFT import Learner
    elif name == "linearprobe":
        from models_CL.linearprobe import Learner
    elif name == "lpft_efm":
        from models_CL.LPFT_EFM import Learner


    elif name == "aper_finetune":
        from models_CL.aper_finetune import Learner
    elif name == "aper_ssf":
        from models_CL.aper_ssf import Learner
    elif name == "aper_vpt":
        from models_CL.aper_vpt import Learner 
    elif name == "aper_adapter":
        from models_CL.aper_adapter import Learner
    elif name == "l2p":
        from models_CL.l2p import Learner
    elif name == "dualprompt":
        from models_CL.dualprompt import Learner
    elif name == "coda_prompt":
        from models_CL.coda_prompt import Learner
        
    


    elif name == "icarl":
        from models_CL.icarl import Learner
    elif name == "der":
        from models_CL.der import Learner
    elif name == "coil":
        from models_CL.coil import Learner
    elif name == "foster":
        from models_CL.foster import Learner
    elif name == "memo":
        from models_CL.memo import Learner
    elif name == 'ranpac':
        from models_CL.ranpac import Learner
    elif name == "ease":
        from models_CL.ease import Learner
    elif name == 'slca':
        from models_CL.slca import Learner
    elif name == 'lae':
        from models_CL.lae import Learner
    elif name == 'fecam':
        from models_CL.fecam import Learner
    elif name == 'dgr':
        from models_CL.dgr import Learner
    elif name == 'mos':
        from models_CL.mos import Learner
    elif name == 'cofima':
        from models_CL.cofima import Learner
    elif name == 'duct':
        from models_CL.duct import Learner
    elif name == 'tuna':
        from models_CL.tuna import Learner
    elif name == 'tuna_efm':
        from models_EFM.tuna_efm import Learner

    else:
        assert 0
    return Learner(args)
