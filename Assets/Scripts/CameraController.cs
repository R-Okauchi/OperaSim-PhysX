using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;        // Float64MultiArrayMsg
using System;
using System.Collections;
using System.IO;

/// <summary>
/// 仮想カメラリグの Unity 追従コンポーネント（論文/README 撮影用・観測専用）。
///
/// cap-pangaea が唯一の camera pose master となり、ROS2 /viz/camera_rig
/// (std_msgs/Float64MultiArray) で正典 pose を配る。本コンポーネントはそれを購読し、
/// ROS/world フレーム → Unity フレームへ逆変換して Main Camera を追従させる。
///
/// 配線: data = [px,py,pz, tx,ty,tz, ux,uy,uz, fov_deg, capture_token] (ROS/world フレーム)。
/// 逆変換 (GroundTruthPublisher.cs の逆): Unity = (-ros.y, ros.z, ros.x)。
/// quaternion でなく look-at を受け取り Unity 側で LookRotation する
/// (ハンドネス取り違えを避けるため)。
///
/// 物理/駆動には一切触れない。Main Camera GameObject に Add Component する
/// (FreeFlyCamera と同居でよい。pose 受信中は FreeFlyCamera を自動で無効化する)。
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraController : MonoBehaviour
{
    [Header("ROS Settings")]
    [Tooltip("購読する camera pose トピック。cap 側 CAP_CAMERA_RIG_TOPIC と一致させる。")]
    public string topicName = "/viz/camera_rig";

    [Header("Capture")]
    [Tooltip("ScreenCapture の出力ディレクトリ。空なら環境変数 CAP_UNITY_SHOT_DIR、それも無ければ persistentDataPath。")]
    public string shotDir = "";
    [Tooltip("capture_token 変化後、撮影までに待つフレーム数 (新 pose が描画されてから撮る)。")]
    public int settleFrames = 4;
    [Tooltip("撮影スーパーサンプル倍率 (1=ゲームビュー解像度)。captureWidth/Height>0 の時は無視。")]
    public int superSize = 1;
    [Tooltip("撮影解像度 幅px。>0 なら RenderTexture に camera.Render で撮る (Game ビュー非フォーカスでも撮れる/固定解像度)。0 で従来 ScreenCapture。")]
    public int captureWidth = 1280;
    [Tooltip("撮影解像度 高さpx。")]
    public int captureHeight = 720;

    [Header("Debug")]
    public bool verbose = false;

    private ROSConnection _ros;
    private Camera _cam;
    private Behaviour _freeFly;       // FreeFlyCamera。別 asmdef (既定 Assembly-CSharp) なので型参照せず文字列解決
    private double _lastToken = double.NaN;

    void Start()
    {
        _cam = GetComponent<Camera>();
        // FreeFlyCamera は別アセンブリ (asmdef 無しの Assembly-CSharp) にあり asmdef 側から型参照
        // できないので、文字列で実行時解決する (CS0246 回避)。FreeFlyCamera : MonoBehaviour : Behaviour。
        _freeFly = GetComponent("FreeFlyCamera") as Behaviour;
        if (string.IsNullOrEmpty(shotDir))
        {
            shotDir = Environment.GetEnvironmentVariable("CAP_UNITY_SHOT_DIR") ?? "";
        }
        if (string.IsNullOrEmpty(shotDir))
        {
            shotDir = Application.persistentDataPath;
        }
        _ros = ROSConnection.GetOrCreateInstance();
        _ros.Subscribe<Float64MultiArrayMsg>(topicName, OnCameraRig);
        Debug.Log($"[{nameof(CameraController)}] subscribed {topicName}, shotDir={shotDir}");
    }

    private void OnCameraRig(Float64MultiArrayMsg msg)
    {
        if (msg == null || msg.data == null || msg.data.Length < 10)
        {
            return;
        }
        var d = msg.data;

        // ROS/world → Unity: Unity = (-ros.y, ros.z, ros.x)。点にも方向ベクトルにも適用可。
        Vector3 posU = new Vector3(-(float)d[1], (float)d[2], (float)d[0]);
        Vector3 tgtU = new Vector3(-(float)d[4], (float)d[5], (float)d[3]);
        Vector3 upU  = new Vector3(-(float)d[7], (float)d[8], (float)d[6]);
        float fovDeg = (float)d[9];

        // director がカメラを握る間は手動 free-fly を止める (上書き合戦の回避)。
        if (_freeFly != null && _freeFly.enabled)
        {
            _freeFly.enabled = false;
        }

        Vector3 forward = tgtU - posU;
        if (forward.sqrMagnitude < 1e-9f)
        {
            return;  // pos==target は LookRotation 不能 (cap 側で elevation をクランプ済みのはず)
        }
        transform.position = posU;
        transform.rotation = Quaternion.LookRotation(forward, upU);
        if (fovDeg > 1.0f)
        {
            _cam.fieldOfView = fovDeg;  // Unity は垂直 FOV (pangaea も垂直)
        }
        float dist = forward.magnitude;
        _cam.nearClipPlane = Mathf.Max(0.1f, dist * 0.01f);
        _cam.farClipPlane = Mathf.Max(1000.0f, dist * 4.0f);

        if (verbose)
        {
            Debug.Log($"[{nameof(CameraController)}] pose posU={posU} tgtU={tgtU} fov={fovDeg} token={d[d.Length - 1]}");
        }

        // capture_token (末尾) が変化したら 1 度だけ撮影する。
        double token = d[d.Length - 1];
        if (!token.Equals(_lastToken))
        {
            _lastToken = token;
            if (token > 0.0)
            {
                StartCoroutine(CaptureAfterSettle((long)token));
            }
        }
    }

    private IEnumerator CaptureAfterSettle(long token)
    {
        // yield return null (= 次フレーム) はプレイヤーループで進むので Game ビュー非フォーカスでも止まらない。
        // (WaitForEndOfFrame はビューが描画されないと進まないため使わない。)
        for (int i = 0; i < Mathf.Max(1, settleFrames); i++)
        {
            yield return null;
        }
        string path = Path.Combine(shotDir, $"unity_view_{token}.png");
        if (captureWidth > 0 && captureHeight > 0)
        {
            CaptureToFile(path, captureWidth, captureHeight);  // camera.Render → フォーカス非依存・固定解像度
        }
        else
        {
            ScreenCapture.CaptureScreenshot(path, Mathf.Max(1, superSize));
        }
        Debug.Log($"[{nameof(CameraController)}] screenshot token={token} -> {path}");
    }

    private void CaptureToFile(string path, int w, int h)
    {
        var rt = new RenderTexture(w, h, 24);
        var prevTarget = _cam.targetTexture;
        var prevActive = RenderTexture.active;
        try
        {
            _cam.targetTexture = rt;
            _cam.Render();  // rig pose のまま RT の aspect で明示描画 (Game ビュー非フォーカスでも実行される)
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Destroy(tex);
        }
        finally
        {
            _cam.targetTexture = prevTarget;
            RenderTexture.active = prevActive;
            rt.Release();
            Destroy(rt);
        }
    }
}
