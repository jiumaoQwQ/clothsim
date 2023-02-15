#include "helper_math.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "MeshVao.h"
#include "Shader.h"
#include "Sphere.h"
#include <vector>
#include <iostream>

GLFWwindow *window;

#define spring_Y 3e4
#define dashpot_damping 1e4
#define drag_damping 1

unsigned int N = 128;
float quad_size = 1.0 / N;

float ball_radius = 0.3;
float3 ball_center = make_float3(0, 0, 0);
float dt = 0.04 / N;
int sub_step = (1.0f / 60.0f) / dt;

float3 *x, *v;

int2 *spring_offset;

__global__ void init_kernel(float3 *_x, float3 *_v,
                            int N, float quad_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > N || y > N)
        return;
    _x[x + N * y] = make_float3(x * quad_size - 0.5, 0.6, y * quad_size - 0.5);
    _v[x + N * y] = make_float3(0, 0, 0);
}

__global__ void init_offset(int2 *offset)
{
    int cnt = 0;
    for (int i = -1; i < 2; i++)
    {
        for (int j = -1; j < 2; j++)
        {
            if (!(i == 0 && j == 0))
            {
                offset[cnt] = make_int2(i, j);
                cnt++;
            }
        }
    }
    offset[cnt++] = make_int2(0, 2);
    offset[cnt++] = make_int2(2, 0);
    offset[cnt++] = make_int2(0, -2);
    offset[cnt++] = make_int2(-2, 0);
}

__global__ void gravity_kernel(float3 *_v, float dt, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > N || y > N)
        return;
    _v[x + y * N] += make_float3(0, -9.8, 0) * dt;
}

__global__ void update_position_kernel(float3 *_x, float3 *_v, float dt, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > N || y > N)
        return;

    _x[x + y * N] += _v[x + y * N] * dt;
}

__global__ void spring_kernel(float3 *_x, float3 *_v, int2 *offset,
                              float quad_size, float dt, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > N || y > N)
        return;
    for (int i = 0; i < 12; i++)
    {
        float3 force = make_float3(0, 0, 0);
        int other_x = x + offset[i].x;
        int other_y = y + offset[i].y;
        if (other_x >= 0 && other_x < N && other_y >= 0 && other_y < N)
        {
            float3 x_ij = _x[x + N * y] - _x[other_x + N * other_y];
            float3 v_ij = _v[x + N * y] - _v[other_x + N * other_y];

            float3 d = normalize(x_ij);
            float current_length = norm3df(x_ij.x, x_ij.y, x_ij.z);
            float original_length = quad_size * norm3df(offset[i].x, offset[i].y, 0);
            force += -spring_Y * d * (current_length / original_length - 1);
            force += -dot(v_ij, d) * d * dashpot_damping * quad_size;
        }
        _v[x + N * y] += force * dt;
    }
    _v[x + N * y] *= exp2f(drag_damping * dt);
}

__global__ void collision_kernel(float3 *_x, float3 *_v,
                                 float3 ball_center, float ball_radius,
                                 float dt, int N)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > N || y > N)
        return;

    float3 delta_x = _x[x + N * y] - ball_center;
    float sdf = norm3df(delta_x.x, delta_x.y, delta_x.z) - ball_radius;
    if (sdf <= 0)
    {
        float3 d = normalize(delta_x);
        _v[x + N * y] -= __min(dot(d, _v[x + N * y]), 0.0f) * d;
    }
}

void step()
{
    gravity_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(v, dt, N);
    spring_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(x, v, spring_offset, quad_size, dt, N);

    collision_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(x, v, ball_center, ball_radius, dt, N);

    update_position_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(x, v, dt, N);
}

__host__ void init_glfw()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1920, 1080, "ClothSim", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height)
                                   { glViewport(0, 0, width, height); });
    gladLoadGL(glfwGetProcAddress);
}

int main()
{
    init_glfw();

    Shader shader("./shader/shader.vs", "./shader/shader.fs");
    std::vector<float3> ball_pos;
    std::vector<unsigned int> ball_index;
    make_sphere(ball_pos,ball_index,ball_center,ball_radius);

    MeshVao ball_vao(ball_pos.size(),ball_index.size());
    ball_vao.copyIn(ball_pos.data(),nullptr,nullptr,ball_index.data(),
    ball_pos.size()*sizeof(float3),0,0,ball_index.size()*sizeof(unsigned int));

    cudaMalloc(&x, N * N * sizeof(float3));
    cudaMalloc(&v, N * N * sizeof(float3));
    cudaMalloc(&spring_offset, 12 * sizeof(int2));

    init_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(x, v,
                                                        N, quad_size);
    init_offset<<<1, 1>>>(spring_offset);

    std::vector<unsigned int> index;

    for (unsigned int i = 0; i < N - 1; i++)
    {
        for (unsigned int j = 0; j < N - 1; j++)
        {
            index.push_back(i + j * N);
            index.push_back(i + 1 + j * N);
            index.push_back(i + (j + 1) * N);
            index.push_back(i + 1 + j * N);
            index.push_back(i + (j + 1) * N);
            index.push_back(i + 1 + (j + 1) * N);
        }
    }

    MeshVao vao(N * N, index.size());

    float3 *cpu_x = (float3 *)malloc(N * N * sizeof(float3));

    int cnt = 0;
    while (true)
    {
        if (glfwWindowShouldClose(window))
            exit(0);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        cnt++;
        if (cnt == 128)
        {
            cnt %= 16;
            init_kernel<<<{(N + 7) / 8, (N + 7) / 8}, {8, 8}>>>(x, v,
                                                                N, quad_size);
        }
        for (int i = 0; i < sub_step; i++)
        {
            step();
        }
        cudaMemcpy(cpu_x, x, N * N * sizeof(float3), cudaMemcpyDeviceToHost);

        vao.copyIn(cpu_x, nullptr, nullptr, index.data(),
                   N * N * sizeof(float3), 0, 0, index.size() * sizeof(unsigned int));
        shader.use();
        shader.setMat4("Project", glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f));
        shader.setMat4("View", glm::lookAt(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)));
        vao.draw();
        ball_vao.draw();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    free(cpu_x);

    cudaFree(x);
    cudaFree(v);
    return 0;
}