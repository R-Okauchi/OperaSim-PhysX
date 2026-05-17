using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.BuiltinInterfaces;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;
using Unity.Robotics.Core;

/// <summary>
/// RGBカメラシミュレーション。Unity Cameraの描画結果をsensor_msgs/Imageとして送信。
/// sensor.mdで「必須」とされるRGBカメラ4-6台のうち1台を実装。
/// マシンのキャビン前方に付ける。
/// </summary>
public class CameraImagePublisher : MonoBehaviour
{
    ROSConnection ros;

    [Tooltip("カメラ画像を出力するROSトピック名")]
    public string topicName = "[robot_name]/camera/front/image_raw";
    private string preprocessedTopicName;

    [Tooltip("フレーム名")]
    public string frameName = "[robot_name]/camera_front_link";
    private string preprocessedFrameName;

    [Tooltip("ROSメッセージの出力間隔(秒)")]
    public float publishMessageInterval = 0.1f; // 10Hz

    [Header("Camera Settings")]
    [Tooltip("画像幅 (pixels)")]
    public int imageWidth = 640;

    [Tooltip("画像高さ (pixels)")]
    public int imageHeight = 480;

    [Tooltip("水平 Field of View (degrees) — Phase 5β-3-cal-2 で SensorSetup.cs から spec ベースで設定。" +
             "0 以下なら Unity Camera default (60°) を維持する。")]
    public float fovDeg = 0f;

    private Camera cam;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private ImageMsg message;
    private float timeElapsed;

    void Start()
    {
        preprocessedTopicName = Utils.PreprocessNamespace(this.gameObject, topicName);
        preprocessedFrameName = Utils.PreprocessNamespace(this.gameObject, frameName);

        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(preprocessedTopicName);

        // Setup camera
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            cam = gameObject.AddComponent<Camera>();
        }
        cam.enabled = false; // We render manually
        // Phase 5β-3-cal-2: apply FOV from inspector / SensorSetup-pushed
        // value. Skip when fovDeg <= 0 so the Unity default (60°) stays
        // in effect for cameras that haven't been calibrated yet.
        if (fovDeg > 0f)
        {
            cam.fieldOfView = fovDeg;
        }

        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        message = new ImageMsg();
        message.header = new HeaderMsg();
        message.header.stamp = new TimeMsg();
        message.height = (uint)imageHeight;
        message.width = (uint)imageWidth;
        message.encoding = "rgb8";
        message.is_bigendian = 0;
        message.step = (uint)(imageWidth * 3); // 3 bytes per pixel (RGB)
    }

    void Update()
    {
        timeElapsed += Time.deltaTime;

        if (timeElapsed >= publishMessageInterval)
        {
            // Render camera to texture
            cam.targetTexture = renderTexture;
            cam.Render();

            // Read pixels
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = null;
            cam.targetTexture = null;

            // Get raw pixel data (RGB24)
            byte[] imageData = texture2D.GetRawTextureData();

            // Unity renders upside down compared to ROS convention, flip vertically
            byte[] flippedData = new byte[imageData.Length];
            int rowSize = imageWidth * 3;
            for (int y = 0; y < imageHeight; y++)
            {
                System.Array.Copy(
                    imageData, y * rowSize,
                    flippedData, (imageHeight - 1 - y) * rowSize,
                    rowSize
                );
            }

            message.header.frame_id = preprocessedFrameName;
            message.header.stamp = new TimeStamp(Clock.time);
            message.data = flippedData;

            ros.Publish(preprocessedTopicName, message);
            timeElapsed = 0.0f;
        }
    }

    void OnDestroy()
    {
        if (renderTexture != null)
        {
            renderTexture.Release();
            Destroy(renderTexture);
        }
        if (texture2D != null)
        {
            Destroy(texture2D);
        }
    }
}
