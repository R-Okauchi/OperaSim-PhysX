//Do not edit! This file was generated by Unity-ROS MessageGeneration.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

namespace RosMessageTypes.Std
{
    [Serializable]
    public class Float32Msg : Message
    {
        public const string k_RosMessageName = "std_msgs/Float32";
        public override string RosMessageName => k_RosMessageName;

        public float data;

        public Float32Msg()
        {
            this.data = 0.0f;
        }

        public Float32Msg(float data)
        {
            this.data = data;
        }

        public static Float32Msg Deserialize(MessageDeserializer deserializer) => new Float32Msg(deserializer);

        private Float32Msg(MessageDeserializer deserializer)
        {
            deserializer.Read(out this.data);
        }

        public override void SerializeTo(MessageSerializer serializer)
        {
            serializer.Write(this.data);
        }

        public override string ToString()
        {
            return "Float32Msg: " +
            "\ndata: " + data.ToString();
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
#else
        [UnityEngine.RuntimeInitializeOnLoadMethod]
#endif
        public static void Register()
        {
            MessageRegistry.Register(k_RosMessageName, Deserialize);
        }
    }
}