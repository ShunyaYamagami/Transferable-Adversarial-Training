cd  ~/lab/gda/da/Transferable-Adversarial-Training/pytorch
# . cmd.sh  0 0 2_4_6_8_10_12_14  -p DomainNet --task true_domains  --tmux TAT_0
# . cmd.sh  1 0 1_3_5_7_9_11_13  -p DomainNet --task true_domains  --tmux TAT_1

. cmd.sh  0 0 2_4_6_8_10_12_14  -p DomainNet --task simclr_encoder_bs512_ep2000_lr0.001_outd64_g3  --tmux TAT_0
. cmd.sh  1 0 3_5_7_9_11_13  -p DomainNet --task simclr_encoder_bs512_ep2000_lr0.001_outd64_g3  --tmux TAT_1

# . cmd.sh  0 0 0_2_4_6_8_10_12_14  -p DomainNet --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep3000_outd64_g3  --tmux TAT_0
# . cmd.sh  1 0 1_3_5_7_9_11_13  -p DomainNet --task contrastive_rpl_dim128_wght0.6_AE_bs512_ep3000_outd64_g3  --tmux TAT_1
