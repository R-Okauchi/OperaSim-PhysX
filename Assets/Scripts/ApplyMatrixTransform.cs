using UnityEngine;
using System.IO;   // For File I/O
using System.Globalization; // For parsing floats

public class ApplyMatrixTransform : MonoBehaviour
{
    [Tooltip("Relative path in Assets or absolute path to the transform_{i}.txt file.")]
    public string transformFilePath = "/Users/okauchiryota/main_desk/UTokyo/llm_safety/valid_data/transform_100.txt";
    
    void Start()
    {
        // Load the 4x4 matrix T from file
        Matrix4x4 T = LoadMatrix4x4(transformFilePath);

        // Decompose T into translation, rotation, and scale
        Vector3 position = ExtractTranslation(T);
        Quaternion rotation = ExtractRotation(T);
        Vector3 scale = ExtractScale(T);

        // Apply to this GameObject’s transform
        transform.localPosition = position;
        transform.localRotation = rotation;
        transform.localScale    = scale;
    }

    /// <summary>
    /// Loads a 4x4 matrix from a text file with 4 lines of 4 floating-point values.
    /// </summary>
    Matrix4x4 LoadMatrix4x4(string path)
    {
        // Read all lines
        string[] lines = File.ReadAllLines(path);

        // We expect 4 lines, each with 4 floats
        float[,] values = new float[4,4];

        for (int i = 0; i < 4; i++)
        {
            // Split by whitespace
            string[] tokens = lines[i].Split(new char[] { ' ', '\t' }, System.StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < 4; j++)
            {
                // Parse with invariant culture (avoid locale issues with commas)
                values[i, j] = float.Parse(tokens[j], CultureInfo.InvariantCulture);
            }
        }

        // Construct the Matrix4x4 (Unity uses m00, m01, ... row-major in memory)
        Matrix4x4 M = new Matrix4x4();
        M.m00 = values[0,0]; M.m01 = values[0,1]; M.m02 = values[0,2]; M.m03 = values[0,3];
        M.m10 = values[1,0]; M.m11 = values[1,1]; M.m12 = values[1,2]; M.m13 = values[1,3];
        M.m20 = values[2,0]; M.m21 = values[2,1]; M.m22 = values[2,2]; M.m23 = values[2,3];
        M.m30 = values[3,0]; M.m31 = values[3,1]; M.m32 = values[3,2]; M.m33 = values[3,3];

        return M;
    }

    /// <summary>
    /// Extracts translation from a 4x4 matrix (last column).
    /// </summary>
    Vector3 ExtractTranslation(Matrix4x4 m)
    {
        return new Vector3(m.m03, m.m13, m.m23);
    }

    /// <summary>
    /// Extract the scale from a 4x4 matrix by measuring the length of basis vectors.
    /// </summary>
    Vector3 ExtractScale(Matrix4x4 m)
    {
        // Each column (m.m00..m.m20, etc.) represents a basis vector for the rotation & scale.
        // The length of that column is the scale for that axis (assuming no shear).
        Vector3 xBasis = new Vector3(m.m00, m.m10, m.m20);
        Vector3 yBasis = new Vector3(m.m01, m.m11, m.m21);
        Vector3 zBasis = new Vector3(m.m02, m.m12, m.m22);

        float scaleX = xBasis.magnitude;
        float scaleY = yBasis.magnitude;
        float scaleZ = zBasis.magnitude;

        return new Vector3(scaleX, scaleY, scaleZ);
    }

    /// <summary>
    /// Extract the rotation quaternion from a 4x4 matrix.
    /// 
    /// The rotation is taken from the upper-left 3x3 part after removing any scale.
    /// </summary>
    Quaternion ExtractRotation(Matrix4x4 m)
    {
        // Remove scale from matrix to isolate pure rotation
        Vector3 scale = ExtractScale(m);

        // Guard against division by zero if scale is zero on any axis:
        if (Mathf.Abs(scale.x) < 1e-5f) scale.x = 1e-5f;
        if (Mathf.Abs(scale.y) < 1e-5f) scale.y = 1e-5f;
        if (Mathf.Abs(scale.z) < 1e-5f) scale.z = 1e-5f;

        Matrix4x4 rotMatrix = new Matrix4x4();

        rotMatrix.m00 = m.m00 / scale.x;
        rotMatrix.m01 = m.m01 / scale.y;
        rotMatrix.m02 = m.m02 / scale.z;
        rotMatrix.m10 = m.m10 / scale.x;
        rotMatrix.m11 = m.m11 / scale.y;
        rotMatrix.m12 = m.m12 / scale.z;
        rotMatrix.m20 = m.m20 / scale.x;
        rotMatrix.m21 = m.m21 / scale.y;
        rotMatrix.m22 = m.m22 / scale.z;

        // The last row/column of a rotation matrix is [0,0,0,1] in ideal math
        rotMatrix.m03 = 0f; rotMatrix.m13 = 0f; rotMatrix.m23 = 0f;
        rotMatrix.m30 = 0f; rotMatrix.m31 = 0f; rotMatrix.m32 = 0f; rotMatrix.m33 = 1f;

        // Convert the 3x3 rotation matrix to Quaternion
        return QuaternionFromMatrix(rotMatrix);
    }

    /// <summary>
    /// Convert a pure rotation (3x3 portion) Matrix4x4 to a Quaternion.
    /// </summary>
    Quaternion QuaternionFromMatrix(Matrix4x4 m)
    {
        // Based on https://forum.unity.com/threads/how-to-decompose-a-trs-matrix.104500/#post-2300932
        // or other standard conversions from rotation matrix to quaternion.
        return Quaternion.LookRotation(
            m.GetColumn(2), // forward
            m.GetColumn(1)  // up
        );
    }
}
