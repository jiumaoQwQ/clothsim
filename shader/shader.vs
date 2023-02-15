#version 330 core
layout (location = 0) in vec3 aPos;    
layout (location = 1) in vec3 aNormal; 
layout (location = 1) in vec3 aTex; 

uniform mat4 Project;
uniform mat4 View;

void main()
{
    gl_Position = Project*View*vec4(aPos, 1.0);
}