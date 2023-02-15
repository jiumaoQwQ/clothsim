#pragma once

#include <vector>
#include <cmath>

void make_sphere(std::vector<float3>& pos ,std::vector<unsigned int>& index ,float3 ball_center, float ball_radius)
{
    const int Y_SEGMENTS = 50;
    const int X_SEGMENTS = 50;
    const float PI = 3.14159;
    for (int y = 0; y <= Y_SEGMENTS; y++)
    {
        for (int x = 0; x <= X_SEGMENTS; x++)
        {
            float xSegment = (float)x / (float)X_SEGMENTS;
            float ySegment = (float)y / (float)Y_SEGMENTS;
            float xPos = ball_radius * std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            float yPos = ball_radius * std::cos(ySegment * PI);
            float zPos = ball_radius * std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
            pos.push_back(ball_center + make_float3(xPos, yPos, zPos));
        }
    }

    for (int i = 0; i < Y_SEGMENTS; i++)
    {
        for (int j = 0; j < X_SEGMENTS; j++)
        {
            index.push_back(i * (X_SEGMENTS + 1) + j);
            index.push_back((i + 1) * (X_SEGMENTS + 1) + j);
            index.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            index.push_back(i * (X_SEGMENTS + 1) + j);
            index.push_back((i + 1) * (X_SEGMENTS + 1) + j + 1);
            index.push_back(i * (X_SEGMENTS + 1) + j + 1);
        }
    }

}