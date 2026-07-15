# ******************************************************************************
#  pyorbbecsdk 初級サンプル 03 — カラーとデプスのアライメント（位置合わせ）ストリーム
#
#  このサンプルで学べること:
#    1. カラーストリームとデプスストリームを同時に有効化する方法
#    2. ソフトウェアの AlignFilter を使って、デプスをカラーカメラの視点に投影する方法
#    3. パイプライン設定でハードウェア D2C（Depth to Color）アライメントを有効化する方法（--hw フラグ）
#    4. 生の uint16 デプスデータをカラーマップ画像に変換する方法
#    5. アライメント済みのデプスとカラーをブレンドして目視確認する方法
#
#  キー操作（ソフトウェアモード、デフォルト）:
#    T       — アライメント方向を切り替え: Depth→Color (D2C) ↔ Color→Depth (C2D)
#    F       — ハードウェアのフレーム同期を ON / OFF 切り替え
#    Q / ESC — 終了
#
#  キー操作（ハードウェア D2C モード、--hw）:
#    T       — ハードウェア D2C を ON / OFF 切り替え
#    + / -   — デプスオーバーレイの透明度を調整
#    Q / ESC — 終了
#
#  依存パッケージ: numpy, opencv-python, utils.py
#
#  対応デバイス: すべて
#
#  実行方法:
#    python examples/beginner/03_color_and_depth_aligned.py          # ソフトウェアアライメント（デフォルト）
#    python examples/beginner/03_my_color_and_depth.py     # ハードウェア D2C
# ******************************************************************************

import math
import argparse
import os
import sys
import time
from ultralytics import YOLO

# utils.py を読み込むために、一つ上の階層をパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import mediapipe as mp
import socket
from utils import frame_to_bgr_image

from pyorbbecsdk import OBAlignMode  # type: ignore
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face Meshの初期化
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,      # 虹彩を含む478点
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

HOST = "127.0.0.1"      # C++サーバーが同じPCならこれ
PORT = 50000

client = None  # main() 内で必要な場合のみ接続する

# 

# --- 設定用の定数 ---
ESC_KEY = 27
MIN_DEPTH = 20  # 有効なデプス距離の最小値（mm）
MAX_DEPTH = 10000  # 有効なデプス距離の最大値（mm）

# YOLOモデルの読み込み
#YOLOmodel = YOLO("yolov11n-pose.pt")  # yolov


# ---------------------------------------------------------------------------
# ハードウェア D2C 用のヘルパー関数（--hw 指定時のみ使用）
# ---------------------------------------------------------------------------

def euclidean_distance(p1, p2):
    """2つの3D座標(cm)間のユークリッド距離(cm)を計算"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def pixel_to_world(u, v, depth_mm, fx, fy, cx, cy):
    """
    u, v      : ピクセル座標（反転していない元画像上）
    depth_mm  : そのピクセルの深度値（mm）
    戻り値    : (X, Y, Z) メートル単位
    """
    Z = depth_mm / 10.0  # mm → m
    if Z <= 0:
        return None
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


def get_hw_stream_config(pipeline: Pipeline):
    """
    ハードウェア D2C アライメントを有効化する Config を構築する。

    カラープロファイルを順番に調べ、RGB フォーマットかつ
    対応するハードウェアアライメント済みデプスプロファイルが
    存在するものを見つけたら、OBAlignMode.HW_MODE を設定する。
    成功時は Config を返し、ハードウェア D2C が未対応の場合は None を返す。
    """
    config = Config()
    try:
        # カラーセンサーで利用可能なストリームプロファイル一覧を取得
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        for i in range(len(profile_list)):
            color_profile = profile_list[i]
            # RGB フォーマット以外はスキップ
            if color_profile.get_format() != OBFormat.RGB:
                continue
            # このカラープロファイルに対応するハードウェアD2Cデプスプロファイルを取得
            hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_depth_list) == 0:
                continue
            # デプスとカラーの両方のストリームを有効化
            config.enable_stream(hw_depth_list[0])
            config.enable_stream(color_profile)
            # ハードウェアD2Cモードを設定
            config.set_align_mode(OBAlignMode.HW_MODE)
            return config
    except Exception as e:
        print(f"HW D2C config error: {e}")
    return None


def switch_hw_d2c(pipeline: Pipeline, config: Config, enable: bool):
    """パイプラインを停止し、ハードウェアD2Cのアライメントモードを切り替えて再開する。"""
    pipeline.stop()
    time.sleep(0.1)  # 停止処理が完了するまで少し待機
    config.set_align_mode(OBAlignMode.HW_MODE if enable else OBAlignMode.DISABLE)
    print(f"Hardware D2C: {'Enabled' if enable else 'Disabled'}")
    pipeline.start(config)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------


def main():
    
    # デバイスが接続されているか確認
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("Device Not Found! Please connect an Orbbec camera and try again.")
        return

    # コマンドライン引数の設定（--hw フラグでハードウェアD2Cモードに切り替え）
    parser = argparse.ArgumentParser(description="Color + Depth aligned viewer")
    parser.add_argument(
        "--hw",
        action="store_true",
        help="Use hardware D2C alignment instead of software AlignFilter",
    )
    parser.add_argument(
        "--no-tcp",
        action="store_true",
        help="Do not connect to the TCP server (run standalone without sending eye data)",
    )
    args = parser.parse_args()

    use_tcp = not args.no_tcp

    global client
    if use_tcp:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("接続中...")
        try:
            client.connect((HOST, PORT))
            print("接続成功")
        except (ConnectionRefusedError, OSError) as e:
            print(f"TCP接続に失敗しました: {e}")
            print("--no-tcp オプションを付けて再実行するか、C++サーバーを先に起動してください。")
            return
    else:
        print("TCPなしモードで起動します（送信は行いません）")
    # 表示用ウィンドウを作成
    window_name = "Color + Depth Aligned  |  Q/ESC = quit"
    window_name_mediapipe = "Color `+ Depth Aligned  |  Q/ESC = quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.namedWindow(window_name_mediapipe, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name_mediapipe, 1280, 720)

    pipeline = Pipeline()
    config = None

    # ------------------------------------------------------------------
    # ストリーム設定 — ソフトウェアモードとハードウェアモードで処理が異なる
    # ------------------------------------------------------------------
    if args.hw:
        # ---- ハードウェア D2C モード ----
        try:
            # ハードウェアのフレーム同期を有効化
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"Frame sync warning: {e}")

        # ハードウェアD2C用のストリーム設定を取得
        config = get_hw_stream_config(pipeline)
        if config is None:
            print("ERROR: Hardware D2C is not supported on this device. Try without --hw.")
            return

        enable_hw_d2c = True
        alpha = 0.5       # オーバーレイの透明度（初期値）
        alpha_step = 0.1  # +/- キーで変化させるステップ幅

        print("\n========== Hardware D2C Align ==========")
        print("T       : Enable / Disable HW D2C")
        print("+ / -   : Adjust depth overlay transparency")
        print("Q / ESC : Quit")
        print("========================================\n")

    else:
        # ---- ソフトウェア AlignFilter モード ----
        config = Config()
        align_mode = 0  # 0 = D2C（デプス→カラー）, 1 = C2D（カラー→デプス）
        enable_sync = False

        try:
            # カラーセンサーのプロファイル一覧を取得し、RGBフォーマットのストリームを有効化
            profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            # 利用可能なプロファイルを確認したい場合
            for i in range(len(profile_list)):
                p = profile_list[i].as_video_stream_profile()
                print(p.get_width(), p.get_height(), p.get_fps(), p.get_format())
            color_profile = profile_list.get_video_stream_profile(1280,720,OBFormat.RGB,30)
            config.enable_stream(color_profile)
            # color_profile を有効化した直後あたりに追加
            intrinsic = color_profile.get_intrinsic()
            fx = intrinsic.fx
            fy = intrinsic.fy
            cx = intrinsic.cx
            cy = intrinsic.cy
            print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            # デプスセンサーのデフォルトプロファイルを有効化
            profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(depth_profile)

            # カラー・デプス両方のフレームが揃うまで待って出力するモードに設定
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        except Exception as e:
            print(f"Stream configuration error: {e}")
            return

        # ソフトウェアアライメントフィルタを作成（デプスをカラー視点に合わせる）
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        print("\nControls:")
        print("  T       — Toggle align direction  (D2C ↔ C2D)")
        print("  F       — Toggle frame sync        (ON / OFF)")
        print("  Q / ESC — Quit\n")

    # ------------------------------------------------------------------
    # パイプライン開始
    # ------------------------------------------------------------------
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return

    # ------------------------------------------------------------------
    # フレーム取得ループ
    # ------------------------------------------------------------------
    while True:
        try:
            # 最大1000msフレームを待機して取得
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # -- ソフトウェアアライメント処理（SWモードのみ） --
            if not args.hw:
                frames = align_filter.process(frames)
                if not frames:
                    continue
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

            # -- カラーフレームをBGR画像に変換 --
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue
            
            # -- デプスフレームをnumpy配列に変換 --
            try:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
            except ValueError:
                continue

            # デプススケールを掛けて実距離（mm）に変換
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

            if args.hw:
                # ハードウェアモードでは範囲外の値をクリップ（切り詰め）
                depth_data = np.clip(depth_data, MIN_DEPTH, MAX_DEPTH)
            else:
                # ソフトウェアモードでは範囲外の値を0（無効値）にする
                depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)

            # デプス値を0〜255に正規化し、カラーマップ（JET）を適用して可視化
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)

            # カラー画像とサイズが異なる場合はリサイズ（ハードウェアD2C無効時などに必要）
            h, w = color_image.shape[:2]
            if depth_image.shape[:2] != (h, w):
                depth_image = cv2.resize(depth_image, (w, h), interpolation=cv2.INTER_NEAREST)

            # -- カラー画像とデプス画像をブレンドして表示 --
            blend_alpha = alpha if args.hw else 0.5
            overlay = cv2.addWeighted(color_image, 1 - blend_alpha, depth_image, blend_alpha, 0)

            # ステータステキストの作成
            if args.hw:
                status = f"HW D2C: {'ON' if enable_hw_d2c else 'OFF'}  alpha={blend_alpha:.1f}"
            else:
                mode_str = "D2C (Depth To Color)" if align_mode == 0 else "C2D (Color To Depth)"
                sync_str = "Sync: ON" if enable_sync else "Sync: OFF"
                status = f"{mode_str} | {sync_str}"

            # 画面左上にステータス情報を描画
            cv2.putText(
                color_image,
                status,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            #cv2.imshow(window_name, color_image)

            #YOLOモデルで姿勢推定
            # results = YOLOmodel(frame)

            # annotated = results[0].plot()

            # MediaPipe処理
            frame = color_image.copy()

            # 左右反転（鏡表示）
            frame = cv2.flip(frame, 1)

            # BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 顔検出
            results = face_mesh.process(rgb)

            # ランドマーク描画
            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:

                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=
                        mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                left_eye = face_landmarks.landmark[468]
                right_eye = face_landmarks.landmark[473]

                h, w, _ = frame.shape

            # 反転後(表示用)の座標
                left_x = int(left_eye.x * w)
                left_y = int(left_eye.y * h)
                right_x = int(right_eye.x * w)
                right_y = int(right_eye.y * h)
                cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
                # 反転前(深度画像と対応する)座標に戻す
                left_x_orig = w - 1 - left_x
                right_x_orig = w - 1 - right_x

                # 深度値を取得（depth_dataは反転前のオリジナル画像に対応）
                left_depth = depth_data[left_y, left_x_orig]
                right_depth = depth_data[right_y, right_x_orig]

                left_world = pixel_to_world(left_x_orig, left_y, left_depth, fx, fy, cx, cy)
                right_world = pixel_to_world(right_x_orig, right_y, right_depth, fx, fy, cx, cy)
                
                if left_world and right_world:
    
                    # 両目の中点を実世界座標(メートル)で計算
                    eye_center_world = tuple((l + r) / 2 for l, r in zip(left_world, right_world))
                    Xw, Yw, Zw = eye_center_world
                    ipd_cm = euclidean_distance(left_world, right_world)

                    # C++側にメートル単位で送信
                    message = f"{Xw:.4f},{Yw:.4f},{Zw:.4f},{ipd_cm:.2f}\n"
                    print(f"Xw: {Xw:.4f}, Yw: {Yw:.4f}, Zw: {Zw:.4f}, IPD: {ipd_cm:.2f} cm")
                    if use_tcp and client is not None:
                        client.sendall(message.encode())
            # cv2.imshow(window_name_mediapipe, frame)
            # -- キーボード入力の処理 --
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ESC_KEY):
                break  # Q または ESC で終了
            if args.hw:
                if key in (ord("t"), ord("T")):
                    # ハードウェアD2Cの有効/無効を切り替え
                    enable_hw_d2c = not enable_hw_d2c
                    switch_hw_d2c(pipeline, config, enable_hw_d2c)
                elif key in (ord("+"), ord("=")):
                    # 透明度を上げる
                    alpha = min(1.0, alpha + alpha_step)
                    print(f"Alpha: {alpha:.2f}")
                elif key in (ord("-"), ord("_")):
                    # 透明度を下げる
                    alpha = max(0.0, alpha - alpha_step)
                    print(f"Alpha: {alpha:.2f}")
            else:
                if key in (ord("t"), ord("T")):
                    # アライメント方向（D2C ↔ C2D）を切り替え
                    align_mode = (align_mode + 1) % 2
                    if align_mode == 0:
                        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                        print("Mode: Depth To Color")
                    else:
                        align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
                        print("Mode: Color To Depth")
                elif key in (ord("f"), ord("F")):
                    # フレーム同期のON/OFFを切り替え
                    enable_sync = not enable_sync
                    if enable_sync:
                        pipeline.enable_frame_sync()
                        print("Frame sync: ON")
                    else:
                        pipeline.disable_frame_sync()
                        print("Frame sync: OFF")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Runtime error: {e}")
            continue

    # 終了処理：ウィンドウを閉じてパイプラインを停止
    cv2.destroyAllWindows()
    pipeline.stop()
    if client is not None:
        client.close()

if __name__ == "__main__":
    main()