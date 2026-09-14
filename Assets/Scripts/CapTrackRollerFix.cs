using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CapKit
{
    /// <summary>
    /// Runtime-only physics fix for the crawler prefabs: rigid non-wheel colliders that sit within a few
    /// centimetres of the wheel contact plane (zx200's six track-roller "Cylinder" meshes at 2-5 cm,
    /// mst110cr's vessel rods at 0-2 cm) act as skids on any terrain undulation. Live 2026-09-15 04:30:
    /// zx200 wedged solid against a 0.16 m rise at the plateau foot (0.0 m in 12 s under a 0.5 m/s
    /// command) and moved again the moment those rollers stopped colliding with the terrain. A real
    /// ZX200 climbs 35 deg. Nothing is edited in the prefabs: collisions between those colliders and the
    /// terrain are ignored per pair after the scene loads, and the result is logged per machine.
    /// </summary>
    public sealed class CapTrackRollerFix : MonoBehaviour
    {
        /// <summary>Colliders whose bounds bottom is this close (m) to the wheel contact plane are skids.</summary>
        public float skidClearanceM = 0.10f;
        /// <summary>Seconds after load before the first pass (machines settle onto their suspension).</summary>
        public float firstPassDelayS = 3f;
        /// <summary>A second pass catches machines that spawn late; 0 disables it.</summary>
        public float secondPassDelayS = 15f;

        private readonly HashSet<int> _done = new HashSet<int>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Install()
        {
            if (FindObjectOfType<CapTrackRollerFix>() != null) return;
            var go = new GameObject("CapTrackRollerFix");
            go.hideFlags = HideFlags.DontSave;
            go.AddComponent<CapTrackRollerFix>();
        }

        private IEnumerator Start()
        {
            yield return new WaitForSeconds(firstPassDelayS);
            Apply();
            if (secondPassDelayS > 0f)
            {
                yield return new WaitForSeconds(secondPassDelayS);
                Apply();
            }
        }

        private void Apply()
        {
            var terrains = FindObjectsOfType<TerrainCollider>();
            if (terrains.Length == 0) return;
            foreach (var drive in FindObjectsOfType<DiffDriveController>())
            {
                var root = drive.transform.root.gameObject;
                if (_done.Contains(root.GetInstanceID())) continue;
                var wheels = root.GetComponentsInChildren<WheelCollider>();
                if (wheels.Length == 0) continue;
                float wheelBottom = float.PositiveInfinity;
                foreach (var w in wheels) wheelBottom = Mathf.Min(wheelBottom, w.transform.position.y - w.radius);
                int ignored = 0;
                var names = new List<string>();
                foreach (var c in root.GetComponentsInChildren<Collider>())
                {
                    if (c is WheelCollider || c.isTrigger || !c.enabled) continue;
                    if (c.bounds.min.y - wheelBottom > skidClearanceM) continue;
                    bool anyPair = false;
                    foreach (var t in terrains)
                    {
                        if (Physics.GetIgnoreLayerCollision(c.gameObject.layer, t.gameObject.layer)) continue;
                        Physics.IgnoreCollision(c, t, true);
                        anyPair = true;
                    }
                    if (!anyPair) continue;
                    ignored++;
                    if (names.Count < 3) names.Add(c.name);
                }
                _done.Add(root.GetInstanceID());
                if (ignored > 0)
                    Debug.Log(string.Format("[CAP Kit] {0}: {1} skid collider(s) within {2:F2} m of the wheel contact plane no longer collide with the terrain ({3})",
                                            root.name, ignored, skidClearanceM, string.Join(", ", names.ToArray())));
            }
        }
    }
}
