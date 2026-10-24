using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace CapKit.Editor
{
    /// <summary>
    /// A prefab's make-up as text: its objects (name, layer, tag, active, local pose) and their components with the
    /// fields that decide how a machine moves (bodies, joints and their drives, wheels, colliders, the scripts'
    /// serialized fields) - to build a new machine the way the existing ones are built, without the editor's window:
    ///   Unity -batchmode -quit -projectPath ... -executeMethod CapKit.Editor.CapPrefabDump.Run
    /// env CAP_DUMP_PREFABS (comma-separated asset paths), CAP_DUMP_OUT (the text file; default Temp/cap_prefab_dump.txt).
    /// </summary>
    public static class CapPrefabDump
    {
        public static void Run()
        {
            string list = Environment.GetEnvironmentVariable("CAP_DUMP_PREFABS") ?? "";
            string output = Environment.GetEnvironmentVariable("CAP_DUMP_OUT");
            if (string.IsNullOrWhiteSpace(output)) output = "Temp/cap_prefab_dump.txt";
            var sb = new StringBuilder();
            foreach (var path in list.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0))
            {
                var root = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                sb.AppendLine($"===== {path}");
                if (root == null)
                {
                    sb.AppendLine("  (not found)");
                    continue;
                }
                Dump(root.transform, 0, sb);
            }
            File.WriteAllText(output, sb.ToString());
            Debug.Log($"[CapPrefabDump] wrote {output}");
        }

        static void Dump(Transform t, int depth, StringBuilder sb)
        {
            string pad = new string(' ', 2 * depth);
            var go = t.gameObject;
            sb.AppendLine($"{pad}- {go.name} [layer {go.layer} {LayerMask.LayerToName(go.layer)}, tag {go.tag}" +
                          $"{(go.activeSelf ? "" : ", inactive")}] pos {V(t.localPosition)} rot {V(t.localEulerAngles)}");
            foreach (var c in go.GetComponents<Component>())
            {
                if (c == null)
                {
                    sb.AppendLine($"{pad}    * (missing script)");
                    continue;
                }
                if (c is Transform) continue;
                sb.AppendLine($"{pad}    * {c.GetType().Name}{Describe(c)}");
                if (c is MonoBehaviour mb) Fields(mb, pad + "        ", sb);
            }
            foreach (Transform child in t) Dump(child, depth + 1, sb);
        }

        static string Describe(Component c)
        {
            switch (c)
            {
                case Rigidbody rb:
                    return $": mass {rb.mass} drag {rb.drag}/{rb.angularDrag} kinematic {rb.isKinematic} " +
                           $"constraints {rb.constraints} com {V(rb.centerOfMass)}";
                case ArticulationBody ab:
                    var d = ab.xDrive;
                    return $": {ab.jointType} mass {ab.mass} immovable {ab.immovable} enabled {ab.enabled} " +
                           $"anchor {V(ab.anchorPosition)}/{V(ab.anchorRotation.eulerAngles)} " +
                           $"drive k {d.stiffness} c {d.damping} f {d.forceLimit} [{d.lowerLimit}, {d.upperLimit}] " +
                           $"target {d.target} twist {ab.twistLock} swingY {ab.swingYLock} swingZ {ab.swingZLock} " +
                           $"linX {ab.linearLockX} com {V(ab.centerOfMass)} auto-com {ab.automaticCenterOfMass}";
                case WheelCollider w:
                    var f = w.forwardFriction;
                    var s = w.sidewaysFriction;
                    var sp = w.suspensionSpring;
                    return $": radius {w.radius} mass {w.mass} centre {V(w.center)} suspension {w.suspensionDistance} " +
                           $"spring {sp.spring} damper {sp.damper} target {sp.targetPosition} fwd stiff {f.stiffness} " +
                           $"side stiff {s.stiffness} enabled {w.enabled}";
                case BoxCollider b:
                    return $": centre {V(b.center)} size {V(b.size)} trigger {b.isTrigger} enabled {b.enabled}";
                case MeshCollider m:
                    return $": convex {m.convex} mesh {(m.sharedMesh ? m.sharedMesh.name : "-")} trigger {m.isTrigger} " +
                           $"enabled {m.enabled}";
                case Collider other:
                    return $": trigger {other.isTrigger} enabled {other.enabled}";
                case FixedJoint fj:
                    return $": connected body {(fj.connectedBody ? fj.connectedBody.name : "-")} articulation " +
                           $"{(fj.connectedArticulationBody ? fj.connectedArticulationBody.name : "-")} mass scale " +
                           $"{fj.massScale}/{fj.connectedMassScale}";
                case MeshFilter mf:
                    return $": {(mf.sharedMesh ? mf.sharedMesh.name : "-")}";
                default:
                    return "";
            }
        }

        static void Fields(MonoBehaviour mb, string pad, StringBuilder sb)
        {
            var so = new SerializedObject(mb);
            var p = so.GetIterator();
            bool enter = true;
            while (p.NextVisible(enter))
            {
                enter = false;
                if (p.name == "m_Script") continue;
                sb.AppendLine($"{pad}{p.name} = {Value(p)}");
            }
        }

        static string Value(SerializedProperty p)
        {
            switch (p.propertyType)
            {
                case SerializedPropertyType.Integer: return p.intValue.ToString();
                case SerializedPropertyType.Boolean: return p.boolValue.ToString();
                case SerializedPropertyType.Float: return p.floatValue.ToString("G6");
                case SerializedPropertyType.String: return $"\"{p.stringValue}\"";
                case SerializedPropertyType.Enum:
                    return p.enumValueIndex >= 0 && p.enumValueIndex < p.enumDisplayNames.Length
                        ? p.enumDisplayNames[p.enumValueIndex] : p.enumValueIndex.ToString();
                case SerializedPropertyType.ObjectReference:
                    return p.objectReferenceValue ? $"{p.objectReferenceValue.name} ({p.objectReferenceValue.GetType().Name})" : "null";
                case SerializedPropertyType.Vector3: return V(p.vector3Value);
                case SerializedPropertyType.Vector2: return p.vector2Value.ToString();
                case SerializedPropertyType.Generic:
                    if (p.isArray)
                    {
                        var items = Enumerable.Range(0, Math.Min(p.arraySize, 12))
                            .Select(i => Value(p.GetArrayElementAtIndex(i)));
                        return $"[{p.arraySize}] {string.Join(", ", items)}";
                    }
                    return "{…}";
                default: return p.propertyType.ToString();
            }
        }

        static string V(Vector3 v) => $"({v.x:G4}, {v.y:G4}, {v.z:G4})";
    }
}
