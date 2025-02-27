using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using UnityEngine;

public class SimpleCmdVelPublisher : MonoBehaviour
{
    // RosConnection.prefab の設定がシーンにある前提
    public string cmdVelTopic = "/zx120/tracks/cmd_vel";

    void Update()
    {
        // 簡単に、Wキーで前進、Aキーで左旋回など
        float linearX = 0.0f;
        float angularZ = 0.0f;

        if (Input.GetKey(KeyCode.W))
            Debug.Log("W key is pressed");
            linearX = 0.5f;
        if (Input.GetKey(KeyCode.S))
            linearX = -0.3f;
        if (Input.GetKey(KeyCode.A))
            angularZ = 0.5f;
        if (Input.GetKey(KeyCode.D))
            angularZ = -0.5f;

        // Twistメッセージを作ってPublish
        TwistMsg msg = new TwistMsg(
            new Vector3Msg(linearX, 0, 0),
            new Vector3Msg(0, 0, angularZ)
        );
        ROSConnection.instance.Publish(cmdVelTopic, msg);
    }
}
