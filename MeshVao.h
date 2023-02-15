#pragma once

#include <glad/gl.h>

struct MeshVao
{
    MeshVao(unsigned int N, unsigned int N2)
    {
        glGenVertexArrays(1, vao);
        glGenBuffers(4, vbo);

        glBindVertexArray(vao[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
        glBufferData(GL_ARRAY_BUFFER, N * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(2);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, N2 * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

        num_of_vertices = N;
        num_of_index = N2;
    }

    void copyIn(void *vertex, void *normal, void *texcoord, void *index,
                unsigned int vertex_size, unsigned int normal_size, unsigned int texcoord_size, unsigned int index_size)
    {
        glBindVertexArray(vao[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        void *ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, vertex, vertex_size);
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, normal, normal_size * 3 * sizeof(float));
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
        ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, texcoord, texcoord_size);
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[3]);
        ptr = glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(ptr, index, index_size);
        glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    }

    void draw()
    {
        glBindVertexArray(vao[0]);

        glDrawElements(GL_TRIANGLES, num_of_index, GL_UNSIGNED_INT, 0);
    }

    unsigned int vao[1], vbo[4];
    unsigned int num_of_vertices, num_of_index;
};