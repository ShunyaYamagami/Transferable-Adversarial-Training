function process_args {
    declare -A args
    local gpu_i=$1
    local exec_num=$2
    local dset_num=$3

    # 残りの名前付き引数を解析
    local dataset="office"
    if [ $dataset = 'office' ]; then
        local task=(
            # "original_uda"
            # "true_domains"
            "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep300_g3_shfl"
        )
    elif [ $dataset = 'home' ]; then
        local task=(
            # "original_uda"
            "true_domains"
            "simclr_rpl_uniform_dim512_wght0.5_bs512_ep300_g3_encoder_outdim64_shfl"
            "simclr_bs512_ep300_g3_shfl"
        )
    fi

    echo "gpu_i: $gpu_i"
    echo "exec_num: $exec_num"
    echo "dset_num: $dset_num"
    echo -e ''

    ##### データセット設定
    if [ $dataset = 'office' ]; then
        dsetlist=("amazon_dslr" "webcam_amazon" "dslr_webcam")
    elif [ $dataset = 'home' ]; then
        dsetlist=("Art_Clipart" "Art_Product" "Art_RealWorld" "Clipart_Product" "Clipart_RealWorld" "Product_RealWorld")
    elif [ $dataset = 'DomainNet' ]; then
        dsetlist=('clipart_infograph' 'clipart_painting' 'clipart_quickdraw' 'clipart_real' 'clipart_sketch' 'infograph_painting' 'infograph_quickdraw' 'infograph_real' 'infograph_sketch' 'painting_quickdraw' 'painting_real' 'painting_sketch' 'quickdraw_real' 'quickdraw_sketch' 'real_sketch')
    else
        echo "不明なデータセット: $dataset" >&2
        return 1
    fi

    COMMAND=''
    COMMAND+="conda deactivate && conda deactivate"
    COMMAND+=" && conda activate tvt"

    for tsk in "${task[@]}"; do
        if [ $dset_num -eq -1 ]; then
            for dset in "${dsetlist[@]}"; do
                # 事前にこれを実行する必要がある．
                COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i python extract_features.py \
                    --dataset $dataset \
                    --task $tsk \
                    --dset $dset \
                    --train_batch_size 512"
                    
                COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i python office.py \
                    --dataset $dataset \
                    --task $tsk \
                    --dset $dset \
                    --train_batch_size 512 "
            done
        else
            COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i python extract_features.py \
                --dataset $dataset \
                --task $tsk \
                --dset $dset \
                --train_batch_size 512"

            COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i python office.py \
                --dataset $dataset \
                --task $tsk \
                --dset $dset \
                --train_batch_size 512 "
        fi
    done

    ###### 実行. 
    echo $COMMAND
    echo ''
    eval $COMMAND
}

# 最初の3つの引数をチェック
if [ "$#" -lt 1 ]; then
    echo "エラー: 引数が足りません。最初の1つの引数は必須です。" >&2
    return 1
fi
########## Main ##########
process_args "$@"