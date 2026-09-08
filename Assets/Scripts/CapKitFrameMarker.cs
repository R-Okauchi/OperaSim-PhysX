using UnityEngine;

namespace CapKit
{
    [DisallowMultipleComponent]
    public sealed class CapKitFrameMarker : MonoBehaviour
    {
        public string sensorId;
        public string model;
        public bool populated;
        public string topic;
        public string frameId;

        // Ownership metadata only; this component performs no runtime work.
        [HideInInspector] public bool managedByCapKit;
        [HideInInspector] public GameObject sensorInstance;
    }
}
