title Train(F) Diff-SVC
SETLOCAL ENABLEDELAYEDEXPANSION

REM ================================
REM root 는 anaconda3의 설치 경로를 입력해줍니다
REM ================================

set root=C:\ProgramData\anaconda3
set dpath=I:\_Diff-svc

REM ================================
REM dpath 는 Diff-svc의 설치 경로를 입력해줍니다
REM ================================


set cpath=%dpath%\data\


if not exist "%dpath%\preprocess\*.wav" ( 
    if not exist "%dpath%\preprocess\*.mp3" ( 
        echo 학습시킬 음원파일이 preprocess 폴더 안에 존재하지 않습니다
        echo 사용법을 숙지해주세요
        start chrome.exe --incognito "https://github.com/wlsdml1114/diff-svc"
        goto :file_not_exist
    )
) else (
REM echo exist
)


:stt1
cls
echo 학습시킬 모델의 이름을 입력해주세요^!
echo (띄어쓰기, 특수문자를 사용하면 안됩니다)
set /p user_name= 이름 입력 후 엔터 : 
if "%user_name%" == "" ( goto :stt1 )
set user_name=%user_name: =%
set user_name=%user_name:\=%
set user_name=%user_name:/=%
set user_name=%user_name::=%
set user_name=%user_name:?=%
set user_name=%user_name:"=%
set user_name=%user_name:<=%
set user_name=%user_name:>=%
set user_name=%user_name:|=%

REM echo %user_name%
set ui_binary=data/binary/%user_name%
set ui_raw_data_dir=data/dataset/%user_name%
set ui_speaker_id=%user_name%
set ui_work_dir=checkpoints/%user_name%
echo.


:stt2
echo 최대로 보존할 CKPT(학습파일) 개수를 입력해주세요 (기본값 10, 최대 100)
set /p user_ckpt= 숫자 입력 후 엔터 : 
if "%user_ckpt%" == "" (
set user_ckpt=10
echo.
goto :stt3
) else (
for /L %%a in (10,1,100) do (
    if "%user_ckpt%" == "%%a" (
        set user_ckpt=%%a
        echo.
        goto :stt3
    )
))
echo.
echo 올바른 값을 입력해주세요
echo.
goto :stt2


:stt3
echo 모델이 한번에 학습할 batch 양을 입력해주세요 (기본값 8, 최대 128)
set /p user_max_sentences= 숫자 입력 후 엔터 : 
if "%user_max_sentences%" == "" (
set user_max_sentences=8
echo.
goto :stt4
) else (
for /L %%a in (8,1,128) do (
    if "%user_max_sentences%" == "%%a" (
        set user_max_sentences=%%a
        echo.
        goto :stt4
    )
))
echo.
echo 올바른 값을 입력해주세요
echo.
goto :stt3


:stt4
echo endless_ds 값을 설정합니다
echo 데이터셋의 전체길이가 1시간 이상인가요^?
echo     1시간보다 길다면, 1 을 입력 후 엔터
echo     1시간보다 짧다면, 0 을 입력 후 엔터
set /p ui_endless_ds= 입력 해주세요 : 
for /L %%a in (0,1,1) do (
    if "%ui_endless_ds%" == "%%a" (
        set ui_endless_ds=%%a
        goto :stt0
    )
)
echo.
echo 올바른 값을 입력해주세요
echo.
goto :stt4
echo.


:stt0
cls
echo 모델 이름 : %user_name%
echo 보존할 학습파일 개수 : %user_ckpt%
if "%ui_endless_ds%" == "0" (
    set ui_endless_ds=True
    echo 데이터셋 길이 : 1시간 이하
) else (
    set ui_endless_ds=False
    echo 데이터셋 길이 : 1시간 이상
)
echo batch size : %user_max_sentences%
echo.
echo 입력한 값이 맞나요^? 
echo     y 입력 후 엔터 누르면 다음으로 진행
echo     n 입력 후 엔터 누르면 처음부터 다시 입력
set /p qs=  입력해주세요 : 
if "%qs%" == "y" (
    goto :sttz
) else (
    if "%qs%" == "n" (    goto :stt1 )
    goto :stt0
)


:sttz
call :write_yaml
echo write_yaml comple
echo.


:Cok
call %root%\Scripts\activate.bat %root%
call cd /d %dpath%
if not exist "%root%\envs\diff-svc\" (
    call conda create -n diff-svc python=3.9
)
call conda activate diff-svc
if not exist "%root%\envs\diff-svc\Lib\site-packages\torch\" (
    call pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
)
if not exist "%root%\envs\diff-svc\Lib\site-packages\torchcrepe\" (
    call pip install -r requirements.txt
)
call set PYTHONPATH=.
call set CUDA_VISIBLE_DEVICES=0
call python sep_wav.py
if not exist "%dpath%\%ui_raw_data_dir%\" (
    call md "%dpath%\%ui_raw_data_dir%"
)
call move /y "%dpath%\preprocess_out\voice\*.*" "%dpath%\%ui_raw_data_dir%"
call move /y "%dpath%\preprocess_out\final\*.*" "%dpath%\%ui_raw_data_dir%"
call rd /s /q "%dpath%\preprocess\"
call rd /s /q "%dpath%\preprocess_out\"
call md "%dpath%\preprocess"
call md "%dpath%\preprocess_out"
call echo ^* > "%dpath%\preprocess\.gitignore"
call echo ^!.gitignore >> "%dpath%\preprocess\.gitignore"
call python preprocessing/binarize.py --config training/config_nsf.yaml
call start chrome.exe --incognito "http://localhost:6006/#scalars&amp;_smoothingWeight=0.999"
call start cmd /C tensorboard --logdir "%dpath%\checkpoints\%user_name%\lightning_logs\lastest"
call python run.py --config training/config_nsf.yaml --exp_name %user_name% --reset
endlocal
rundll32 user32.dll,MessageBeep
exit


:file_not_exist
endlocal
rundll32 user32.dll,MessageBeep
pause
exit


:write_yaml
echo write_yaml
echo # setting for users> training/config_nsf.yaml
echo ## original wav dataset folder >> training/config_nsf.yaml
echo raw_data_dir: %ui_raw_data_dir% >> training/config_nsf.yaml
echo ## after binarized dataset folder >> training/config_nsf.yaml
echo binary_data_dir: %ui_binary% >> training/config_nsf.yaml
echo ## speaker name >> training/config_nsf.yaml
echo speaker_id: %ui_speaker_id% >> training/config_nsf.yaml
echo ## trained model will be save this folder >> training/config_nsf.yaml
echo work_dir: %ui_work_dir% >> training/config_nsf.yaml
echo ## batch size >> training/config_nsf.yaml
echo max_sentences: %user_max_sentences% >> training/config_nsf.yaml
echo ## AMP(Automatic Mixed Precision) setting(only GPU) for less VRAM >> training/config_nsf.yaml
echo use_amp: true >> training/config_nsf.yaml
echo. >> training/config_nsf.yaml
echo # setting for developers and advanced users >> training/config_nsf.yaml
echo K_step: 1000 >> training/config_nsf.yaml
echo accumulate_grad_batches: 1 >> training/config_nsf.yaml
echo audio_num_mel_bins: 128 >> training/config_nsf.yaml
echo audio_sample_rate: 44100 >> training/config_nsf.yaml
echo binarization_args: >> training/config_nsf.yaml
echo   shuffle: false >> training/config_nsf.yaml
echo   with_align: true >> training/config_nsf.yaml
echo   with_f0: true >> training/config_nsf.yaml
echo   with_hubert: true >> training/config_nsf.yaml
echo   with_spk_embed: false >> training/config_nsf.yaml
echo   with_wav: false >> training/config_nsf.yaml
echo binarizer_cls: preprocessing.SVCpre.SVCBinarizer >> training/config_nsf.yaml
echo check_val_every_n_epoch: 10 >> training/config_nsf.yaml
echo choose_test_manually: false >> training/config_nsf.yaml
echo clip_grad_norm: 1 >> training/config_nsf.yaml
echo config_path: training/config_nsf.yaml >> training/config_nsf.yaml
echo content_cond_steps: [] >> training/config_nsf.yaml
echo cwt_add_f0_loss: false >> training/config_nsf.yaml
echo cwt_hidden_size: 128 >> training/config_nsf.yaml
echo cwt_layers: 2 >> training/config_nsf.yaml
echo cwt_loss: l1 >> training/config_nsf.yaml
echo cwt_std_scale: 0.8 >> training/config_nsf.yaml
echo datasets: >> training/config_nsf.yaml
echo - opencpop >> training/config_nsf.yaml
echo debug: false >> training/config_nsf.yaml
echo dec_ffn_kernel_size: 9 >> training/config_nsf.yaml
echo dec_layers: 4 >> training/config_nsf.yaml
echo decay_steps: 40000 >> training/config_nsf.yaml
echo decoder_type: fft >> training/config_nsf.yaml
echo dict_dir: '' >> training/config_nsf.yaml
echo diff_decoder_type: wavenet >> training/config_nsf.yaml
echo diff_loss_type: l2 >> training/config_nsf.yaml
echo dilation_cycle_length: 4 >> training/config_nsf.yaml
echo dropout: 0.1 >> training/config_nsf.yaml
echo ds_workers: 4 >> training/config_nsf.yaml
echo dur_enc_hidden_stride_kernel: >> training/config_nsf.yaml
echo - 0,2,3 >> training/config_nsf.yaml
echo - 0,2,3 >> training/config_nsf.yaml
echo - 0,1,3 >> training/config_nsf.yaml
echo dur_loss: mse >> training/config_nsf.yaml
echo dur_predictor_kernel: 3 >> training/config_nsf.yaml
echo dur_predictor_layers: 5 >> training/config_nsf.yaml
echo enc_ffn_kernel_size: 9 >> training/config_nsf.yaml
echo enc_layers: 4 >> training/config_nsf.yaml
echo encoder_K: 8 >> training/config_nsf.yaml
echo encoder_type: fft >> training/config_nsf.yaml
echo endless_ds: %ui_endless_ds% >> training/config_nsf.yaml
echo f0_bin: 256 >> training/config_nsf.yaml
echo f0_max: 1100.0 >> training/config_nsf.yaml
echo f0_min: 40.0 >> training/config_nsf.yaml
echo ffn_act: gelu >> training/config_nsf.yaml
echo ffn_padding: SAME >> training/config_nsf.yaml
echo fft_size: 2048 >> training/config_nsf.yaml
echo fmax: 16000 >> training/config_nsf.yaml
echo fmin: 40 >> training/config_nsf.yaml
echo fs2_ckpt: '' >> training/config_nsf.yaml
echo gaussian_start: true >> training/config_nsf.yaml
echo gen_dir_name: '' >> training/config_nsf.yaml
echo gen_tgt_spk_id: -1 >> training/config_nsf.yaml
echo hidden_size: 256 >> training/config_nsf.yaml
echo hop_size: 512 >> training/config_nsf.yaml
echo hubert_path: checkpoints/hubert/hubert_soft.pt >> training/config_nsf.yaml
echo hubert_gpu: true >> training/config_nsf.yaml
echo infer: false >> training/config_nsf.yaml
echo keep_bins: 128 >> training/config_nsf.yaml
echo lambda_commit: 0.25 >> training/config_nsf.yaml
echo lambda_energy: 0.0 >> training/config_nsf.yaml
echo lambda_f0: 1.0 >> training/config_nsf.yaml
echo lambda_ph_dur: 0.3 >> training/config_nsf.yaml
echo lambda_sent_dur: 1.0 >> training/config_nsf.yaml
echo lambda_uv: 1.0 >> training/config_nsf.yaml
echo lambda_word_dur: 1.0 >> training/config_nsf.yaml
echo load_ckpt: '' >> training/config_nsf.yaml
echo log_interval: 100 >> training/config_nsf.yaml
echo loud_norm: false >> training/config_nsf.yaml
echo lr: 0.0008 >> training/config_nsf.yaml
echo max_beta: 0.02 >> training/config_nsf.yaml
echo max_epochs: 3000 >> training/config_nsf.yaml
echo max_eval_sentences: 1 >> training/config_nsf.yaml
echo max_eval_tokens: 60000 >> training/config_nsf.yaml
echo max_frames: 42000 >> training/config_nsf.yaml
echo max_input_tokens: 60000 >> training/config_nsf.yaml
echo max_tokens: 128000 >> training/config_nsf.yaml
echo max_updates: 1000000 >> training/config_nsf.yaml
echo mel_loss: ssim:0.5^|l1:0.5 >> training/config_nsf.yaml
echo mel_vmax: 1.5 >> training/config_nsf.yaml
echo mel_vmin: -6.0 >> training/config_nsf.yaml
echo min_level_db: -120 >> training/config_nsf.yaml
echo norm_type: gn >> training/config_nsf.yaml
echo num_ckpt_keep: %user_ckpt% >> training/config_nsf.yaml
echo num_heads: 2 >> training/config_nsf.yaml
echo num_sanity_val_steps: 1 >> training/config_nsf.yaml
echo num_spk: 1 >> training/config_nsf.yaml
echo num_test_samples: 0 >> training/config_nsf.yaml
echo num_valid_plots: 10 >> training/config_nsf.yaml
echo optimizer_adam_beta1: 0.9 >> training/config_nsf.yaml
echo optimizer_adam_beta2: 0.98 >> training/config_nsf.yaml
echo out_wav_norm: false >> training/config_nsf.yaml
echo pe_ckpt: checkpoints/0102_xiaoma_pe/model_ckpt_steps_60000.ckpt >> training/config_nsf.yaml
echo pe_enable: false >> training/config_nsf.yaml
echo perform_enhance: true >> training/config_nsf.yaml
echo pitch_ar: false >> training/config_nsf.yaml
echo pitch_enc_hidden_stride_kernel: >> training/config_nsf.yaml
echo - 0,2,5 >> training/config_nsf.yaml
echo - 0,2,5 >> training/config_nsf.yaml
echo - 0,2,5 >> training/config_nsf.yaml
echo pitch_extractor: parselmouth >> training/config_nsf.yaml
echo pitch_loss: l2 >> training/config_nsf.yaml
echo pitch_norm: log >> training/config_nsf.yaml
echo pitch_type: frame >> training/config_nsf.yaml
echo pndm_speedup: 10 >> training/config_nsf.yaml
echo pre_align_args: >> training/config_nsf.yaml
echo   allow_no_txt: false >> training/config_nsf.yaml
echo   denoise: false >> training/config_nsf.yaml
echo   forced_align: mfa >> training/config_nsf.yaml
echo   txt_processor: zh_g2pM >> training/config_nsf.yaml
echo   use_sox: true >> training/config_nsf.yaml
echo   use_tone: false >> training/config_nsf.yaml
echo pre_align_cls: data_gen.singing.pre_align.SingingPreAlign >> training/config_nsf.yaml
echo predictor_dropout: 0.5 >> training/config_nsf.yaml
echo predictor_grad: 0.1 >> training/config_nsf.yaml
echo predictor_hidden: -1 >> training/config_nsf.yaml
echo predictor_kernel: 5 >> training/config_nsf.yaml
echo predictor_layers: 5 >> training/config_nsf.yaml
echo prenet_dropout: 0.5 >> training/config_nsf.yaml
echo prenet_hidden_size: 256 >> training/config_nsf.yaml
echo pretrain_fs_ckpt: '' >> training/config_nsf.yaml
echo processed_data_dir: xxx >> training/config_nsf.yaml
echo profile_infer: false >> training/config_nsf.yaml
echo ref_norm_layer: bn >> training/config_nsf.yaml
echo rel_pos: true >> training/config_nsf.yaml
echo reset_phone_dict: true >> training/config_nsf.yaml
echo residual_channels: 384 >> training/config_nsf.yaml
echo residual_layers: 20 >> training/config_nsf.yaml
echo save_best: false >> training/config_nsf.yaml
echo save_ckpt: true >> training/config_nsf.yaml
echo save_codes: >> training/config_nsf.yaml
echo - configs >> training/config_nsf.yaml
echo - modules >> training/config_nsf.yaml
echo - src >> training/config_nsf.yaml
echo - utils >> training/config_nsf.yaml
echo save_f0: true >> training/config_nsf.yaml
echo save_gt: false >> training/config_nsf.yaml
echo schedule_type: linear >> training/config_nsf.yaml
echo seed: 1234 >> training/config_nsf.yaml
echo sort_by_len: true >> training/config_nsf.yaml
echo spec_max: >> training/config_nsf.yaml
echo - 0.0 >> training/config_nsf.yaml
echo spec_min: >> training/config_nsf.yaml
echo - -5.0 >> training/config_nsf.yaml
echo spk_cond_steps: [] >> training/config_nsf.yaml
echo stop_token_weight: 5.0 >> training/config_nsf.yaml
echo task_cls: training.task.SVC_task.SVCTask >> training/config_nsf.yaml
echo test_ids: [] >> training/config_nsf.yaml
echo test_input_dir: '' >> training/config_nsf.yaml
echo test_num: 0 >> training/config_nsf.yaml
echo test_prefixes: >> training/config_nsf.yaml
echo - test >> training/config_nsf.yaml
echo test_set_name: test >> training/config_nsf.yaml
echo timesteps: 1000 >> training/config_nsf.yaml
echo train_set_name: train >> training/config_nsf.yaml
echo use_crepe: true >> training/config_nsf.yaml
echo use_denoise: false >> training/config_nsf.yaml
echo use_energy_embed: false >> training/config_nsf.yaml
echo use_gt_dur: false >> training/config_nsf.yaml
echo use_gt_f0: false >> training/config_nsf.yaml
echo use_midi: false >> training/config_nsf.yaml
echo use_nsf: true >> training/config_nsf.yaml
echo use_pitch_embed: true >> training/config_nsf.yaml
echo use_pos_embed: true >> training/config_nsf.yaml
echo use_spk_embed: false >> training/config_nsf.yaml
echo use_spk_id: false >> training/config_nsf.yaml
echo use_split_spk_id: false >> training/config_nsf.yaml
echo use_uv: false >> training/config_nsf.yaml
echo use_vec: false >> training/config_nsf.yaml
echo use_var_enc: false >> training/config_nsf.yaml
echo val_check_interval: 2000 >> training/config_nsf.yaml
echo valid_num: 0 >> training/config_nsf.yaml
echo valid_set_name: valid >> training/config_nsf.yaml
echo vocoder: network.vocoders.nsf_hifigan.NsfHifiGAN >> training/config_nsf.yaml
echo vocoder_ckpt: checkpoints/nsf_hifigan/model >> training/config_nsf.yaml
echo warmup_updates: 2000 >> training/config_nsf.yaml
echo wav2spec_eps: 1e-6 >> training/config_nsf.yaml
echo weight_decay: 0 >> training/config_nsf.yaml
echo win_size: 2048 >> training/config_nsf.yaml
echo no_fs2: true >> training/config_nsf.yaml
goto :eof