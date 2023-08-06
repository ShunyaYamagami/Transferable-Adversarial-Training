. cmd.sh  0 0 -1 -p OfficeHome --task true_domains  --tmux  true_domains__simclr
. cmd.sh  1 0 -1 -p OfficeHome --task simclr_rpl_dim128_wght0.5_bs512_ep3000_g3_encoder_outdim64_shfl  --tmux simclr_rpl__simclr

. cmd.sh  0 0 0_1_2 -p OfficeHome --task simclr_bs512_ep1000_g3_shfl  --tmux true_domains__simclr
. cmd.sh  1 0 3_4_5 -p OfficeHome --task simclr_bs512_ep1000_g3_shfl  --tmux simclr_rpl__simclr
