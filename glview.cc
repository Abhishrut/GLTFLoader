#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <GL/glew.h>
#define GLM_ENABLE_EXPERIMENTAL
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include"../common/glm/glm/glm.hpp"
#include"../common/glm/glm/gtc/matrix_inverse.hpp"
#include"../common/glm/glm/gtc/matrix_transform.hpp"
#include"../common/glm/glm/gtc/type_ptr.hpp"
#include"../common/glm/glm/gtx/quaternion.hpp"

#ifdef _WIN32
#include "../common/trackball.h"
#else
#include "trackball.h"
#endif

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef _WIN32
#include "../../tiny_gltf.h"
#else
#include "tiny_gltf.h"
#endif

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

#define CheckGLErrors(desc)                                                   \
  {                                                                           \
    GLenum e = glGetError();                                                  \
    if (e != GL_NO_ERROR) {                                                   \
      printf("OpenGL error in \"%s\": %d (%d) %s:%d\n", desc, e, e, __FILE__, \
             __LINE__);                                                       \
      exit(20);                                                               \
    }                                                                         \
  }

#define CAM_Z (3.0f)
int width = 1920;
int height = 1080;

double prevMouseX, prevMouseY;
bool mouseLeftPressed;
bool mouseMiddlePressed;
bool mouseRightPressed;
float curr_quat[4];
float prev_quat[4];
float eye[3], lookat[3], up[3];
int textureTrue=0;
GLFWwindow *window;

typedef struct {
  GLuint vb;
} GLBufferState;

typedef struct {
  std::vector<GLuint> diffuseTex;  // for each primitive in mesh
} GLMeshState;

typedef struct {
  std::map<std::string, GLint> attribs;
  std::map<std::string, GLint> uniforms;
} GLProgramState;

typedef struct {
  GLuint vb;     // vertex buffer
  size_t count;  // byte count
} GLCurvesState;

std::map<int, GLBufferState> gBufferState;
std::map<std::string, GLMeshState> gMeshState;
std::map<int, GLCurvesState> gCurvesMesh;
GLProgramState gGLProgramState;

struct Node;

struct Material {
  glm::vec4 baseColorFactor = glm::vec4(1.0f);
  uint32_t baseColorTextureIndex;
};

struct Texture {
    int32_t imageIndex;
};

struct Primitive {
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t materialIndex;
};
struct Mesh {
  std::vector<Primitive> primitives;
};

struct Node {
    Node* parent;
    uint32_t index;
    std::vector<Node *> children;
    Mesh mesh;
    glm::vec3 translation {};
    glm::vec3 scale{1.0f};
    glm::quat rotation{};
    int32_t skin =-1;
    glm::mat4 matrix;
    glm::mat4 getLocalMatrix() {
        return glm::translate(glm::mat4(1.0f),translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f),scale)*matrix;
    }
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 color;
    glm::vec4 jointIndices;
    glm::vec4 jointWeights;
};

struct AnimationSampler {
    std::string interpolation;
    std::vector<float> inputs;
    std::vector<glm::vec4> outputsVec4;
};

struct AnimationChannel {
  std::string path;
  Node * Node;
  uint32_t samplerIndex;

};

struct Animation {
    std::string name;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start=std::numeric_limits<float>::max();
    float end= std::numeric_limits<float>::min();
    float currentTime = 0.0f;

};

struct Skin {
    std::string name;
    Node* skeletonRoot = nullptr;
    std::vector<glm::mat4>inverseBindMatrices;
    std::vector<Node*> joints;
};

std::vector<Texture> textures;
std::vector<Material> materials;
std::vector <Node*> nodes;
std::vector<Skin> skins;
std::vector <Animation> animations;
uint32_t activeAnimation =0;


void CheckErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

static std::string GetFilePathExtension(const std::string &FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}

bool LoadShader(GLenum shaderType,  // GL_VERTEX_SHADER or GL_FRAGMENT_SHADER(or
                                    // maybe GL_COMPUTE_SHADER)
                GLuint &shader, const char *shaderSourceFilename) {
  GLint val = 0;

  // free old shader/program
  if (shader != 0) {
    glDeleteShader(shader);
  }

  std::vector<GLchar> srcbuf;
  FILE *fp = fopen(shaderSourceFilename, "rb");
  if (!fp) {
    fprintf(stderr, "failed to load shader: %s\n", shaderSourceFilename);
    return false;
  }
  fseek(fp, 0, SEEK_END);
  size_t len = ftell(fp);
  rewind(fp);
  srcbuf.resize(len + 1);
  len = fread(&srcbuf.at(0), 1, len, fp);
  srcbuf[len] = 0;
  fclose(fp);

  const GLchar *srcs[1];
  srcs[0] = &srcbuf.at(0);

  shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, srcs, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &val);
  if (val != GL_TRUE) {
    char log[4096];
    GLsizei msglen;
    glGetShaderInfoLog(shader, 4096, &msglen, log);
    printf("%s\n", log);
    // assert(val == GL_TRUE && "failed to compile shader");
    printf("ERR: Failed to load or compile shader [ %s ]\n",
           shaderSourceFilename);
    return false;
  }

  printf("Load shader [ %s ] OK\n", shaderSourceFilename);
  return true;
}

bool LinkShader(GLuint &prog, GLuint &vertShader, GLuint &fragShader) {
  GLint val = 0;

  if (prog != 0) {
    glDeleteProgram(prog);
  }

  prog = glCreateProgram();

  glAttachShader(prog, vertShader);
  glAttachShader(prog, fragShader);
  glLinkProgram(prog);

  glGetProgramiv(prog, GL_LINK_STATUS, &val);
  assert(val == GL_TRUE && "failed to link shader");

  printf("Link shader OK\n");

  return true;
}

void reshapeFunc(GLFWwindow *window, int w, int h) {
  (void)window;
  int fb_w, fb_h;
  glfwGetFramebufferSize(window, &fb_w, &fb_h);
  glViewport(0, 0, fb_w, fb_h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, (float)w / (float)h, 0.1f, 1000.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  width = w;
  height = h;
}

void keyboardFunc(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  (void)scancode;
  (void)mods;
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    // Close window
    if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }
}

void clickFunc(GLFWwindow *window, int button, int action, int mods) {
  double x, y;
  glfwGetCursorPos(window, &x, &y);

  bool shiftPressed = (mods & GLFW_MOD_SHIFT);
  bool ctrlPressed = (mods & GLFW_MOD_CONTROL);

  if ((button == GLFW_MOUSE_BUTTON_LEFT) && (!shiftPressed) && (!ctrlPressed)) {
    mouseLeftPressed = true;
    mouseMiddlePressed = false;
    mouseRightPressed = false;
    if (action == GLFW_PRESS) {
      int id = -1;
      // int id = ui.Proc(x, y);
      if (id < 0) {  // outside of UI
        trackball(prev_quat, 0.0, 0.0, 0.0, 0.0);
      }
    } else if (action == GLFW_RELEASE) {
      mouseLeftPressed = false;
    }
  }
  if ((button == GLFW_MOUSE_BUTTON_RIGHT) ||
      ((button == GLFW_MOUSE_BUTTON_LEFT) && ctrlPressed)) {
    if (action == GLFW_PRESS) {
      mouseRightPressed = true;
      mouseLeftPressed = false;
      mouseMiddlePressed = false;
    } else if (action == GLFW_RELEASE) {
      mouseRightPressed = false;
    }
  }
  if ((button == GLFW_MOUSE_BUTTON_MIDDLE) ||
      ((button == GLFW_MOUSE_BUTTON_LEFT) && shiftPressed)) {
    if (action == GLFW_PRESS) {
      mouseMiddlePressed = true;
      mouseLeftPressed = false;
      mouseRightPressed = false;
    } else if (action == GLFW_RELEASE) {
      mouseMiddlePressed = false;
    }
  }
}

void motionFunc(GLFWwindow *window, double mouse_x, double mouse_y) {
  (void)window;
  float rotScale = 1.0f;
  float transScale = 2.0f;

  if (mouseLeftPressed) {
    trackball(prev_quat, rotScale * (2.0f * prevMouseX - width) / (float)width,
              rotScale * (height - 2.0f * prevMouseY) / (float)height,
              rotScale * (2.0f * mouse_x - width) / (float)width,
              rotScale * (height - 2.0f * mouse_y) / (float)height);

    add_quats(prev_quat, curr_quat, curr_quat);
  } else if (mouseMiddlePressed) {
    eye[0] += -transScale * (mouse_x - prevMouseX) / (float)width;
    lookat[0] += -transScale * (mouse_x - prevMouseX) / (float)width;
    eye[1] += transScale * (mouse_y - prevMouseY) / (float)height;
    lookat[1] += transScale * (mouse_y - prevMouseY) / (float)height;
  } else if (mouseRightPressed) {
    eye[2] += transScale * (mouse_y - prevMouseY) / (float)height;
    lookat[2] += transScale * (mouse_y - prevMouseY) / (float)height;
  }

  // Update mouse point
  prevMouseX = mouse_x;
  prevMouseY = mouse_y;
}

static size_t ComponentTypeByteSize(int type) {
  switch (type) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    case TINYGLTF_COMPONENT_TYPE_BYTE:
      return sizeof(char);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    case TINYGLTF_COMPONENT_TYPE_SHORT:
      return sizeof(short);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_INT:
      return sizeof(int);
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
      return sizeof(float);
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
      return sizeof(double);
    default:
      return 0;
  }
}

Node* findNode(Node* parent, uint32_t index) {
    Node* nodeFound=nullptr;
    if (parent->index == index) {
        return parent;
    }
    for (auto& child : parent->children) {
        nodeFound =findNode(child,index);
      if (nodeFound) {
          break;
      }
    }
    return nodeFound;
}

Node* nodeFromIndex(uint32_t index) {
    Node* nodeFound=nullptr;
  for (auto &node : nodes) {
    nodeFound = findNode(node,index);
    if (nodeFound) {
        break;
    }
  }
  return nodeFound;
}

glm::mat4 getNodeMatrix(Node *node) {
  glm::mat4 nodeMatrix = node->getLocalMatrix();
  Node *currentParent = node->parent;
  while (currentParent) {
    nodeMatrix = currentParent->getLocalMatrix() * nodeMatrix;
    currentParent = currentParent->parent;
  }
  return nodeMatrix;
}

void updateJoints(Node *node, GLuint pid) {
  glUseProgram(pid);
  GLuint jointMatrices_u = glGetUniformLocation(pid, "jointMatrices");
  if (node->skin > -1) {
    glm::mat4 inverseTransform = glm::inverse(getNodeMatrix(node));
    Skin skin = skins[node->skin];
    size_t numJoints = (uint32_t)skin.joints.size();
    std::vector<glm::mat4> jointMatrices(numJoints);
    for (size_t i = 0; i < numJoints; i++) {
      jointMatrices[i] =
          getNodeMatrix(skin.joints[i]) * skin.inverseBindMatrices[i];
      jointMatrices[i] = inverseTransform * jointMatrices[i];
    }
    glUniformMatrix4fv(jointMatrices_u, numJoints, GL_FALSE,
                       &jointMatrices[0][0][0]);
  }
  for (auto &child : node->children) {
    updateJoints(child, pid);
  }  
}

void loadSkins(tinygltf::Model &input, GLuint pid) {
  skins.resize(input.skins.size());

  for (size_t i = 0; i < input.skins.size(); i++) {
    tinygltf::Skin glTFSkin = input.skins[i];
    skins[i].name=glTFSkin.name;

    skins[i].skeletonRoot = nodeFromIndex(glTFSkin.skeleton);
      for (int jointIndex : glTFSkin.joints) {
          Node* node = nodeFromIndex(jointIndex);
        if (node) {
            skins[i].joints.push_back(node);
        }
      }

      if (glTFSkin.inverseBindMatrices > -1) {
        const tinygltf::Accessor &accessor =
            input.accessors[glTFSkin.inverseBindMatrices];
        const tinygltf::BufferView &bufferView =
            input.bufferViews[accessor.bufferView];
        const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];

        skins[i].inverseBindMatrices.resize(accessor.count);
        memcpy(skins[i].inverseBindMatrices.data(),
               &buffer.data[accessor.byteOffset + bufferView.byteOffset],
               accessor.count * sizeof(glm::mat4));
        glUseProgram(pid);
        GLuint jointMatrices_u = glGetUniformLocation(pid, "jointMatrices");
        size_t numJoints = (uint32_t)skins[i].joints.size();
        glUniformMatrix4fv(jointMatrices_u, numJoints, GL_FALSE, &skins[i].inverseBindMatrices[0][0][0]);

      }
  
    }
 }

void loadAnimations(tinygltf::Model& input) { 
    animations.resize(input.animations.size());

    for (size_t i = 0; i < input.animations.size(); i++) {
      tinygltf::Animation glTFAnimation = input.animations[i];
      animations[i].name =glTFAnimation.name;
      animations[i].samplers.resize(glTFAnimation.samplers.size());
      for (size_t j = 0; j < glTFAnimation.samplers.size(); j++) {
        tinygltf::AnimationSampler glTFSampler = glTFAnimation.samplers[j];
        AnimationSampler &dstSampler = animations[i].samplers[j];
        dstSampler.interpolation =glTFSampler.interpolation;

        { 
            const tinygltf::Accessor &accessor = input.accessors[glTFSampler.input];
            const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer= input.buffers[bufferView.buffer];

            const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.buffer];

            const float* buf =static_cast<const float*> (dataPtr);

            for (size_t index = 0; index < accessor.count; index++) {
              dstSampler.inputs.push_back(buf[index]);            
            }

            for (auto input : animations[i].samplers[j].inputs) {
              if (input < animations[i].start) {
                animations[i].start =input;
              }
              if (input > animations[i].end) {
                animations[i].end = input;
              }
            }
        }
        {
          const tinygltf::Accessor &accessor =
              input.accessors[glTFSampler.output];
          const tinygltf::BufferView &bufferView =
              input.bufferViews[accessor.bufferView];

          const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];
          const void *dataPtr =
              &buffer.data[accessor.byteOffset + bufferView.byteOffset];

          switch (accessor.type) {
            case TINYGLTF_TYPE_VEC3: {
              const glm::vec3 *buf = static_cast<const glm::vec3 *>(dataPtr);
              for (size_t index = 0; index < accessor.count; index++) {
                dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
              }
              break;
            }
            case TINYGLTF_TYPE_VEC4: {
              const glm::vec4 *buf = static_cast<const glm::vec4 *>(dataPtr);
              for (size_t index = 0; index < accessor.count; index++) {
                dstSampler.outputsVec4.push_back(buf[index]);
              }
              break;
            }
            default: {
              std::cout << "unknown type" << std::endl;
            }
          }
        }
      }

      animations[i].channels.resize(glTFAnimation.channels.size());
      for (size_t j = 0; j < glTFAnimation.channels.size(); j++) {
        tinygltf::AnimationChannel glTFChannel=glTFAnimation.channels[j];
        AnimationChannel &dstChannel = animations[i].channels[j];
        dstChannel.path=glTFChannel.target_path;
        dstChannel.samplerIndex=glTFChannel.sampler;
        dstChannel.Node = nodeFromIndex(glTFChannel.target_node);

      }
    }
}

void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input,
              Node *parent, uint32_t nodeIndex,
              std::vector<uint32_t> &indexBuffer,
              std::vector<Vertex> &vertexBuffer) {
  Node *node = new Node();
  node->parent = parent;
  node->matrix = glm::mat4(1.0f);
  node->index = nodeIndex;
  node->skin = inputNode.skin;

  if (inputNode.translation.size() == 3) {
    node->translation = glm::make_vec3(inputNode.translation.data());
  }

  if (inputNode.rotation.size() == 4) {
    node->rotation = glm::make_quat(inputNode.rotation.data());
  }

  if (inputNode.scale.size() == 3) {
    node->scale=glm::make_vec3(inputNode.scale.data());
  }
  if (inputNode.matrix.size() == 16) {
    node->matrix = glm::make_mat4x4(inputNode.matrix.data());
  }

  if (inputNode.children.size() > 0) {
    for (size_t i = 0; i < inputNode.children.size(); i++) {
      loadNode(input.nodes[inputNode.children[i]],input,node,inputNode.children[i], indexBuffer,vertexBuffer);
    }
  }

  if (inputNode.mesh > -1) {
    const tinygltf::Mesh mesh =input.meshes[inputNode.mesh];
    for (size_t i = 0; i < mesh.primitives.size(); i++) {
        const tinygltf::Primitive& glTFPrimitive = mesh.primitives[i];
        uint32_t firstIndex=static_cast<uint32_t>(indexBuffer.size());
        uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
        uint32_t indexCount=0;
        bool hasSkin = false;
        {
        
            const float* positionBuffer =nullptr;
            const float* normalsBuffer = nullptr;
            const float* texCoordsBuffer = nullptr;
            const uint16_t* jointIndicesBuffer = nullptr;
            const float* jointWeightsBuffer = nullptr;
            size_t vertexCount = 0;

            if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end()) {
              const tinygltf::Accessor &accessor =
                  input.accessors[glTFPrimitive.attributes.find("POSITION")
                                      ->second];
              const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
              positionBuffer = reinterpret_cast<const float *>(&input.buffers[view.buffer].data[accessor.byteOffset+view.byteOffset]);
              vertexCount=accessor.count;
            }

            if (glTFPrimitive.attributes.find("NORMAL") !=
                glTFPrimitive.attributes.end()) {
              const tinygltf::Accessor &accessor =
                  input.accessors[glTFPrimitive.attributes.find("NORMAL")-> second];
              const tinygltf::BufferView& view= input.bufferViews[accessor.bufferView];
              normalsBuffer = reinterpret_cast<const float *>(& input.buffers[view.buffer].data[accessor.byteOffset+view.byteOffset]);
            }

            if (glTFPrimitive.attributes.find("TEXCOORD_0") !=
                glTFPrimitive.attributes.end()) {
              const tinygltf::Accessor &accessor =
                  input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")
                                      ->second];
              const tinygltf::BufferView &view =
                  input.bufferViews[accessor.bufferView];
              texCoordsBuffer = reinterpret_cast<const float *> (&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
            }

            if (glTFPrimitive.attributes.find("JOINTS_0") !=
                glTFPrimitive.attributes.end()) {
              const tinygltf::Accessor &accessor =
                  input.accessors[glTFPrimitive.attributes.find("JOINTS_0")
                                      ->second];
              const tinygltf::BufferView &view =
                  input.bufferViews[accessor.bufferView];
              jointIndicesBuffer = reinterpret_cast<const uint16_t *> (&input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]);
            }

            if (glTFPrimitive.attributes.find("WEIGHTS_0") !=
                glTFPrimitive.attributes.end()) {
              const tinygltf::Accessor &accessor =
                  input.accessors[glTFPrimitive.attributes.find("WEIGHTS_0")
                                      ->second];
              const tinygltf::BufferView &view =
                  input.bufferViews[accessor.bufferView];
              jointWeightsBuffer = reinterpret_cast<const float *>(
                  &input.buffers[view.buffer]
                       .data[accessor.byteOffset + view.byteOffset]);
            }

            hasSkin = (jointIndicesBuffer && jointWeightsBuffer);

            for (size_t v = 0; v < vertexCount; v++) {
                Vertex vert{};
                vert.pos=glm::make_vec3(&positionBuffer[v*3]);
                vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v*3]):glm::vec3(0.0f)));

                vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec2(0.0f);

                vert.color = glm::vec3(1.0f);
                vert.jointIndices =
                    hasSkin
                        ? glm::vec4(glm::make_vec4(&jointIndicesBuffer[v * 4]))
                        : glm::vec4(0.0f);
                vert.jointWeights =
                    hasSkin ? glm::make_vec4(&jointWeightsBuffer[v*4]):glm::vec4(0.0f);
            }
        }

        {
        
            const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.indices];
            const tinygltf::BufferView& bufferView= input.bufferViews[accessor.bufferView];
            const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];

            indexCount=static_cast<uint32_t>(accessor.count);

            switch (accessor.componentType) {
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:{
                  const uint32_t *buf = reinterpret_cast<const uint32_t *>(& buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                  
                  for (size_t index = 0; index < accessor.count; index++) {
                    indexBuffer.push_back(buf[index] +vertexStart);
                  
                  }
                  break;

                }

                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:{
                  const uint16_t *buf = reinterpret_cast<const uint16_t *>(
                      &buffer
                           .data[accessor.byteOffset + bufferView.byteOffset]);

                  for (size_t index = 0; index < accessor.count; index++) {
                    indexBuffer.push_back(buf[index] + vertexStart);
                  }
                  break;
                }
                
                case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:{
                  const uint8_t *buf = reinterpret_cast<const uint8_t *>(
                      &buffer
                           .data[accessor.byteOffset + bufferView.byteOffset]);

                  for (size_t index = 0; index < accessor.count; index++) {
                    indexBuffer.push_back(buf[index] + vertexStart);
                  }
                  break;
                }
                default:
                  std::cerr << " Index component type" << accessor.componentType
                            << " not supported!" << std::endl;
                  return;
            }
        }

        Primitive primitive{};
        primitive.firstIndex = firstIndex;
        primitive.indexCount=indexCount;
        primitive.materialIndex= glTFPrimitive.material;
        node->mesh.primitives.push_back(primitive);

    }
  }

  if (parent) {
    parent->children.push_back(node);
  } else {
  nodes.push_back(node);  
  }
}

void updateAnimation(float deltaTime, GLuint pid) { 
    if (animations.size()==0)return;

    if (activeAnimation > static_cast<uint32_t>(animations.size()) - 1) {
      std::cout << "No animation with index " << activeAnimation << std::endl;
      return;
    }

    Animation& animation= animations[activeAnimation];
    animation.currentTime += deltaTime;
    if (animation.currentTime > animation.end) {
        animation.currentTime -=animation.end;
    }

    for (auto &channel : animation.channels) {
        AnimationSampler& sampler =animation.samplers[channel.samplerIndex];
      for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
          if (sampler.interpolation != "LINEAR") {
            std::cout << " Only linear is supported \n";
            continue;
          }

          if ((animation.currentTime >= sampler.inputs[i]) &&
              (animation.currentTime <= sampler.inputs[i + 1])) {
            float a= (animation.currentTime -sampler.inputs[i])/(sampler.inputs[i+1]-sampler.inputs[i]);

            if (channel.path == "translation") {
              channel.Node->translation = glm::vec3(glm::mix(
                  sampler.outputsVec4[i], sampler.outputsVec4[i+1],a));

            }

            if (channel.path == "rotation") {
              glm::quat q1;
              q1.x = sampler.outputsVec4[i].x;
              q1.y = sampler.outputsVec4[i].y;
              q1.z = sampler.outputsVec4[i].z;
              q1.w = sampler.outputsVec4[i].w;

              glm::quat q2;
              q2.x = sampler.outputsVec4[i+1].x;
              q2.y = sampler.outputsVec4[i+1].y;
              q2.z = sampler.outputsVec4[i+1].z;
              q2.w = sampler.outputsVec4[i+1].w;
            }

            if(channel.path =="scale"){
              channel.Node->scale = glm::vec3(glm::mix(sampler.outputsVec4[i],sampler.outputsVec4[i+1],a));
            }
           }
        }
    } 
    for (auto &node : nodes) {
        updateJoints(node,pid);
    }
}

static void SetupMeshState(tinygltf::Model &model, GLuint progId) {
  // Buffer
  {
    for (size_t i = 0; i < model.bufferViews.size(); i++) {
      const tinygltf::BufferView &bufferView = model.bufferViews[i];
      if (bufferView.target == 0) {
        std::cout << "WARN: bufferView.target is zero" << std::endl;
        continue;  // Unsupported bufferView.
      }

      int sparse_accessor = -1;
      for (size_t a_i = 0; a_i < model.accessors.size(); ++a_i) {
        const auto &accessor = model.accessors[a_i];
        if (accessor.bufferView == i) {
          std::cout << i << " is used by accessor " << a_i << std::endl;
          if (accessor.sparse.isSparse) {
            std::cout
                << "WARN: this bufferView has at least one sparse accessor to "
                   "it. We are going to load the data as patched by this "
                   "sparse accessor, not the original data"
                << std::endl;
            sparse_accessor = a_i;
            break;
          }
        }
      }

      const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
      GLBufferState state;
      glGenBuffers(1, &state.vb);
      glBindBuffer(bufferView.target, state.vb);
      std::cout << "buffer.size= " << buffer.data.size()
                << ", byteOffset = " << bufferView.byteOffset << std::endl;

      if (sparse_accessor < 0)
        glBufferData(bufferView.target, bufferView.byteLength,
                     &buffer.data.at(0) + bufferView.byteOffset,
                     GL_STATIC_DRAW);
      else {
        const auto accessor = model.accessors[sparse_accessor];
        // copy the buffer to a temporary one for sparse patching
        unsigned char *tmp_buffer = new unsigned char[bufferView.byteLength];
        memcpy(tmp_buffer, buffer.data.data() + bufferView.byteOffset,
               bufferView.byteLength);

        const size_t size_of_object_in_buffer =
            ComponentTypeByteSize(accessor.componentType);
        const size_t size_of_sparse_indices =
            ComponentTypeByteSize(accessor.sparse.indices.componentType);

        const auto &indices_buffer_view =
            model.bufferViews[accessor.sparse.indices.bufferView];
        const auto &indices_buffer = model.buffers[indices_buffer_view.buffer];

        const auto &values_buffer_view =
            model.bufferViews[accessor.sparse.values.bufferView];
        const auto &values_buffer = model.buffers[values_buffer_view.buffer];

        for (size_t sparse_index = 0; sparse_index < accessor.sparse.count;
             ++sparse_index) {
          int index = 0;
          // std::cout << "accessor.sparse.indices.componentType = " <<
          // accessor.sparse.indices.componentType << std::endl;
          switch (accessor.sparse.indices.componentType) {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
              index = (int)*(
                  unsigned char *)(indices_buffer.data.data() +
                                   indices_buffer_view.byteOffset +
                                   accessor.sparse.indices.byteOffset +
                                   (sparse_index * size_of_sparse_indices));
              break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
              index = (int)*(
                  unsigned short *)(indices_buffer.data.data() +
                                    indices_buffer_view.byteOffset +
                                    accessor.sparse.indices.byteOffset +
                                    (sparse_index * size_of_sparse_indices));
              break;
            case TINYGLTF_COMPONENT_TYPE_INT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
              index = (int)*(
                  unsigned int *)(indices_buffer.data.data() +
                                  indices_buffer_view.byteOffset +
                                  accessor.sparse.indices.byteOffset +
                                  (sparse_index * size_of_sparse_indices));
              break;
          }
          std::cout << "updating sparse data at index  : " << index
                    << std::endl;
          // index is now the target of the sparse index to patch in
          const unsigned char *read_from =
              values_buffer.data.data() +
              (values_buffer_view.byteOffset +
               accessor.sparse.values.byteOffset) +
              (sparse_index * (size_of_object_in_buffer * accessor.type));

          /*
          std::cout << ((float*)read_from)[0] << "\n";
          std::cout << ((float*)read_from)[1] << "\n";
          std::cout << ((float*)read_from)[2] << "\n";
          */

          unsigned char *write_to =
              tmp_buffer + index * (size_of_object_in_buffer * accessor.type);

          memcpy(write_to, read_from, size_of_object_in_buffer * accessor.type);
        }

        // debug:
        /*for(size_t p = 0; p < bufferView.byteLength/sizeof(float); p++)
        {
          float* b = (float*)tmp_buffer;
          std::cout << "modified_buffer [" << p << "] = " << b[p] << '\n';
        }*/

        glBufferData(bufferView.target, bufferView.byteLength, tmp_buffer,
                     GL_STATIC_DRAW);
        delete[] tmp_buffer;
      }
      glBindBuffer(bufferView.target, 0);

      gBufferState[i] = state;
    }
  }

#if 0  // TODO(syoyo): Implement
	// Texture
	{
		for (size_t i = 0; i < model.meshes.size(); i++) {
			const tinygltf::Mesh &mesh = model.meshes[i];

			gMeshState[mesh.name].diffuseTex.resize(mesh.primitives.size());
			for (size_t primId = 0; primId < mesh.primitives.size(); primId++) {
				const tinygltf::Primitive &primitive = mesh.primitives[primId];

				gMeshState[mesh.name].diffuseTex[primId] = 0;

				if (primitive.material < 0) {
					continue;
				}
				tinygltf::Material &mat = model.materials[primitive.material];
				// printf("material.name = %s\n", mat.name.c_str());
				if (mat.values.find("diffuse") != mat.values.end()) {
					std::string diffuseTexName = mat.values["diffuse"].string_value;
					if (model.textures.find(diffuseTexName) != model.textures.end()) {
						tinygltf::Texture &tex = model.textures[diffuseTexName];
						if (scene.images.find(tex.source) != model.images.end()) {
							tinygltf::Image &image = model.images[tex.source];
							GLuint texId;
							glGenTextures(1, &texId);
							glBindTexture(tex.target, texId);
							glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
							glTexParameterf(tex.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
							glTexParameterf(tex.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

							// Ignore Texture.fomat.
							GLenum format = GL_RGBA;
							if (image.component == 3) {
								format = GL_RGB;
							}
							glTexImage2D(tex.target, 0, tex.internalFormat, image.width,
									image.height, 0, format, tex.type,
									&image.image.at(0));

							CheckErrors("texImage2D");
							glBindTexture(tex.target, 0);

							printf("TexId = %d\n", texId);
							gMeshState[mesh.name].diffuseTex[primId] = texId;
						}
					}
				}
			}
		}
	}
#endif

    
    //meshes are 1
    //103 primitives, 25 materials, 69 textures
    
    //Texture info, 
    
    std::cout << std::endl << "Model data:" << model.materials.size() << " : " <<model.textures.size()<<std::endl;
    for (int i = 0; i < model.meshes[0].primitives.size(); i++) {
    std::cout<< std::endl<< model.materials[model.meshes[0].primitives[i].material].normalTexture.index<< std::endl;
    }
    /*

*/
  glUseProgram(progId);
  GLint vtloc = glGetAttribLocation(progId, "in_vertex");
  GLint nrmloc = glGetAttribLocation(progId, "in_normal");
  GLint uvloc = glGetAttribLocation(progId, "in_texcoord");

  // GLint diffuseTexLoc = glGetUniformLocation(progId, "diffuseTex");
  GLint isCurvesLoc = glGetUniformLocation(progId, "uIsCurve");

  gGLProgramState.attribs["POSITION"] = vtloc;
  gGLProgramState.attribs["NORMAL"] = nrmloc;
  gGLProgramState.attribs["TEXCOORD_0"] = uvloc;
  // gGLProgramState.uniforms["diffuseTex"] = diffuseTexLoc;
  gGLProgramState.uniforms["isCurvesLoc"] = isCurvesLoc;
};

#if 0  // TODO(syoyo): Implement
// Setup curves geometry extension
static void SetupCurvesState(tinygltf::Scene &scene, GLuint progId) {
	// Find curves primitive.
	{
		std::map<std::string, tinygltf::Mesh>::const_iterator it(
				scene.meshes.begin());
		std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(
				scene.meshes.end());

		for (; it != itEnd; it++) {
			const tinygltf::Mesh &mesh = it->second;

			// Currently we only support one primitive per mesh.
			if (mesh.primitives.size() > 1) {
				continue;
			}

			for (size_t primId = 0; primId < mesh.primitives.size(); primId++) {
				const tinygltf::Primitive &primitive = mesh.primitives[primId];

				gMeshState[mesh.name].diffuseTex[primId] = 0;

				if (primitive.material.empty()) {
					continue;
				}

				bool has_curves = false;
				if (primitive.extras.IsObject()) {
					if (primitive.extras.Has("ext_mode")) {
						const tinygltf::Value::Object &o =
							primitive.extras.Get<tinygltf::Value::Object>();
						const tinygltf::Value &ext_mode = o.find("ext_mode")->second;

						if (ext_mode.IsString()) {
							const std::string &str = ext_mode.Get<std::string>();
							if (str.compare("curves") == 0) {
								has_curves = true;
							}
						}
					}
				}

				if (!has_curves) {
					continue;
				}

				// Construct curves buffer
				const tinygltf::Accessor &vtx_accessor =
					scene.accessors[primitive.attributes.find("POSITION")->second];
				const tinygltf::Accessor &nverts_accessor =
					scene.accessors[primitive.attributes.find("NVERTS")->second];
				const tinygltf::BufferView &vtx_bufferView =
					scene.bufferViews[vtx_accessor.bufferView];
				const tinygltf::BufferView &nverts_bufferView =
					scene.bufferViews[nverts_accessor.bufferView];
				const tinygltf::Buffer &vtx_buffer =
					scene.buffers[vtx_bufferView.buffer];
				const tinygltf::Buffer &nverts_buffer =
					scene.buffers[nverts_bufferView.buffer];

				// std::cout << "vtx_bufferView = " << vtx_accessor.bufferView <<
				// std::endl;
				// std::cout << "nverts_bufferView = " << nverts_accessor.bufferView <<
				// std::endl;
				// std::cout << "vtx_buffer.size = " << vtx_buffer.data.size() <<
				// std::endl;
				// std::cout << "nverts_buffer.size = " << nverts_buffer.data.size() <<
				// std::endl;

				const int *nverts =
					reinterpret_cast<const int *>(nverts_buffer.data.data());
				const float *vtx =
					reinterpret_cast<const float *>(vtx_buffer.data.data());

				// Convert to GL_LINES data.
				std::vector<float> line_pts;
				size_t vtx_offset = 0;
				for (int k = 0; k < static_cast<int>(nverts_accessor.count); k++) {
					for (int n = 0; n < nverts[k] - 1; n++) {

						line_pts.push_back(vtx[3 * (vtx_offset + n) + 0]);
						line_pts.push_back(vtx[3 * (vtx_offset + n) + 1]);
						line_pts.push_back(vtx[3 * (vtx_offset + n) + 2]);

						line_pts.push_back(vtx[3 * (vtx_offset + n + 1) + 0]);
						line_pts.push_back(vtx[3 * (vtx_offset + n + 1) + 1]);
						line_pts.push_back(vtx[3 * (vtx_offset + n + 1) + 2]);

						// std::cout << "p0 " << vtx[3 * (vtx_offset + n) + 0] << ", "
						//                  << vtx[3 * (vtx_offset + n) + 1] << ", "
						//                  << vtx[3 * (vtx_offset + n) + 2] << std::endl;

						// std::cout << "p1 " << vtx[3 * (vtx_offset + n+1) + 0] << ", "
						//                  << vtx[3 * (vtx_offset + n+1) + 1] << ", "
						//                  << vtx[3 * (vtx_offset + n+1) + 2] << std::endl;
					}

					vtx_offset += nverts[k];
				}

				GLCurvesState state;
				glGenBuffers(1, &state.vb);
				glBindBuffer(GL_ARRAY_BUFFER, state.vb);
				glBufferData(GL_ARRAY_BUFFER, line_pts.size() * sizeof(float),
						line_pts.data(), GL_STATIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				state.count = line_pts.size() / 3;
				gCurvesMesh[mesh.name] = state;

				// Material
				tinygltf::Material &mat = scene.materials[primitive.material];
				// printf("material.name = %s\n", mat.name.c_str());
				if (mat.values.find("diffuse") != mat.values.end()) {
					std::string diffuseTexName = mat.values["diffuse"].string_value;
					if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
						tinygltf::Texture &tex = scene.textures[diffuseTexName];
						if (scene.images.find(tex.source) != scene.images.end()) {
							tinygltf::Image &image = scene.images[tex.source];
							GLuint texId;
							glGenTextures(1, &texId);
							glBindTexture(tex.target, texId);
							glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
							glTexParameterf(tex.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
							glTexParameterf(tex.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

							// Ignore Texture.fomat.
							GLenum format = GL_RGBA;
							if (image.component == 3) {
								format = GL_RGB;
							}
							glTexImage2D(tex.target, 0, tex.internalFormat, image.width,
									image.height, 0, format, tex.type,
									&image.image.at(0));

							CheckErrors("texImage2D");
							glBindTexture(tex.target, 0);

							printf("TexId = %d\n", texId);
							gMeshState[mesh.name].diffuseTex[primId] = texId;
						}
					}
				}
			}
		}
	}

	glUseProgram(progId);
	GLint vtloc = glGetAttribLocation(progId, "in_vertex");
	GLint nrmloc = glGetAttribLocation(progId, "in_normal");
	GLint uvloc = glGetAttribLocation(progId, "in_texcoord");

	GLint diffuseTexLoc = glGetUniformLocation(progId, "diffuseTex");
	GLint isCurvesLoc = glGetUniformLocation(progId, "uIsCurves");

	gGLProgramState.attribs["POSITION"] = vtloc;
	gGLProgramState.attribs["NORMAL"] = nrmloc;
	gGLProgramState.attribs["TEXCOORD_0"] = uvloc;
	gGLProgramState.uniforms["diffuseTex"] = diffuseTexLoc;
	gGLProgramState.uniforms["uIsCurves"] = isCurvesLoc;
};
#endif

static void DrawMesh(tinygltf::Model &model, const tinygltf::Mesh &mesh) {
  //// Skip curves primitive.
  // if (gCurvesMesh.find(mesh.name) != gCurvesMesh.end()) {
  //  return;
  //}

    /*
  if (gGLProgramState.uniforms["diffuseTex"] >= 0) {
    glUniform1i(gGLProgramState.uniforms["diffuseTex"], 0);  // TEXTURE0
  }
*/
  if (gGLProgramState.uniforms["isCurvesLoc"] >= 0) {
    glUniform1i(gGLProgramState.uniforms["isCurvesLoc"], textureTrue);
  }
  GLuint texid;
  glGenTextures(1, &texid);
      
  int count=0;
  for (size_t i = 0; i < mesh.primitives.size(); i++) {
    const tinygltf::Primitive &primitive = mesh.primitives[i];

    if (primitive.indices < 0) return;

    // Assume TEXTURE_2D target for the texture object.
    //glBindTexture(GL_TEXTURE_2D, gMeshState[mesh.name].diffuseTex[i]);

    //add texture
    /*
    std::cout << std::endl
              << "Model data:" << model.materials.size() << " : "
              << model.textures.size() << std::endl;
    for (int i = 0; i < model.meshes[0].primitives.size(); i++) {
      std::cout << std::endl
                << model.materials[model.meshes[0].primitives[i].material]
                       .normalTexture.index
                << std::endl;
    }
    */
    
    if (model.textures.size() > 0) {
      // fixme: Use material's baseColor
        
       tinygltf::Texture &tex =
            model.textures[model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index];
      //std::cout<<"Tex"<< model.materials[primitive.material].normalTexture.index<< " "<<<<std::endl;

      if (tex.source > -1) {
        
      
        tinygltf::Image &image = model.images[tex.source];

        glActiveTexture(GL_TEXTURE0+count);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texid);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        count++;
        GLenum format = GL_RGBA;

        if (image.component == 1) {
          format = GL_RED;
        } else if (image.component == 2) {
          format = GL_RG;
        } else if (image.component == 3) {
          format = GL_RGB;
        } else {
          // ???
        }

        GLenum type = GL_UNSIGNED_BYTE;
        if (image.bits == 8) {
          // ok
        } else if (image.bits == 16) {
          type = GL_UNSIGNED_SHORT;
        } else {
          // ???
        }
        textureTrue = 1;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0,
                     format, type, &image.image.at(0));

        //glBindTexture(GL_TEXTURE_2D, 0);
        //glActiveTexture(GL_TEXTURE0);  
        //glBindTexture(GL_TEXTURE_2D, 0);  
        //glDisable(GL_TEXTURE_2D);   
      }
    }

    

    std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
    std::map<std::string, int>::const_iterator itEnd(
        primitive.attributes.end());

    for (; it != itEnd; it++) {
      assert(it->second >= 0);
      const tinygltf::Accessor &accessor = model.accessors[it->second];
      glBindBuffer(GL_ARRAY_BUFFER, gBufferState[accessor.bufferView].vb);
      CheckErrors("bind buffer");
      int size = 1;
      if (accessor.type == TINYGLTF_TYPE_SCALAR) {
        size = 1;
      } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
        size = 2;
      } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
        size = 3;
      } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
        size = 4;
      } else {
        assert(0);
      }
      // it->first would be "POSITION", "NORMAL", "TEXCOORD_0", ...
      if ((it->first.compare("POSITION") == 0) ||
          (it->first.compare("NORMAL") == 0) ||
          (it->first.compare("TEXCOORD_0") == 0))
          {
        if (gGLProgramState.attribs[it->first] >= 0) {
          // Compute byteStride from Accessor + BufferView combination.
          int byteStride =
              accessor.ByteStride(model.bufferViews[accessor.bufferView]);
          assert(byteStride != -1);
          glVertexAttribPointer(gGLProgramState.attribs[it->first], size,
                                accessor.componentType,
                                accessor.normalized ? GL_TRUE : GL_FALSE,
                                byteStride, BUFFER_OFFSET(accessor.byteOffset));
          CheckErrors("vertex attrib pointer");
          glEnableVertexAttribArray(gGLProgramState.attribs[it->first]);
          CheckErrors("enable vertex attrib array");
        }
      } else {
        //std::cout << "Hehe" << it->first << std::endl;
            
      }
    }

    const tinygltf::Accessor &indexAccessor =
        model.accessors[primitive.indices];
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
                 gBufferState[indexAccessor.bufferView].vb);
    CheckErrors("bind buffer");
    int mode = -1;
    if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
      mode = GL_TRIANGLES;
    } else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
      mode = GL_TRIANGLE_STRIP;
    } else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
      mode = GL_TRIANGLE_FAN;
    } else if (primitive.mode == TINYGLTF_MODE_POINTS) {
      mode = GL_POINTS;
    } else if (primitive.mode == TINYGLTF_MODE_LINE) {
      mode = GL_LINES;
    } else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
      mode = GL_LINE_LOOP;
    } else {
      assert(0);
    }
    glDrawElements(mode, indexAccessor.count, indexAccessor.componentType,
                   BUFFER_OFFSET(indexAccessor.byteOffset));
        
    CheckErrors("draw elements");

    {
      std::map<std::string, int>::const_iterator it(
          primitive.attributes.begin());
      std::map<std::string, int>::const_iterator itEnd(
          primitive.attributes.end());

      for (; it != itEnd; it++) {
        if ((it->first.compare("POSITION") == 0) ||
            (it->first.compare("NORMAL") == 0) ||
            (it->first.compare("TEXCOORD_0") == 0)) {
          if (gGLProgramState.attribs[it->first] >= 0) {
            glDisableVertexAttribArray(gGLProgramState.attribs[it->first]);
          }
        }
      }
    }
    
  }
  //glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0+count);  
    glBindTexture(GL_TEXTURE_2D, 0);  
    glDisable(GL_TEXTURE_2D);   
    glDeleteTextures(1, &texid);
    
}

#if 0  // TODO(syoyo): Implement
static void DrawCurves(tinygltf::Scene &scene, const tinygltf::Mesh &mesh) {
	(void)scene;

	if (gCurvesMesh.find(mesh.name) == gCurvesMesh.end()) {
		return;
	}

	if (gGLProgramState.uniforms["isCurvesLoc"] >= 0) {
		glUniform1i(gGLProgramState.uniforms["isCurvesLoc"], 1);
	}

	GLCurvesState &state = gCurvesMesh[mesh.name];

	if (gGLProgramState.attribs["POSITION"] >= 0) {
		glBindBuffer(GL_ARRAY_BUFFER, state.vb);
		glVertexAttribPointer(gGLProgramState.attribs["POSITION"], 3, GL_FLOAT,
				GL_FALSE, /* stride */ 0, BUFFER_OFFSET(0));
		CheckErrors("curve: vertex attrib pointer");
		glEnableVertexAttribArray(gGLProgramState.attribs["POSITION"]);
		CheckErrors("curve: enable vertex attrib array");
	}

	glDrawArrays(GL_LINES, 0, state.count);

	if (gGLProgramState.attribs["POSITION"] >= 0) {
		glDisableVertexAttribArray(gGLProgramState.attribs["POSITION"]);
	}
}
#endif

// Hierarchically draw nodes
static void DrawNode(tinygltf::Model &model, const tinygltf::Node &node) {
  // Apply xform

  glPushMatrix();
  if (node.matrix.size() == 16) {
    // Use `matrix' attribute
    glMultMatrixd(node.matrix.data());
  } else {
    // Assume Trans x Rotate x Scale order
    if (node.scale.size() == 3) {
      glScaled(node.scale[0], node.scale[1], node.scale[2]);
    }

    if (node.rotation.size() == 4) {
      glRotated(node.rotation[0], node.rotation[1], node.rotation[2],
                node.rotation[3]);
    }

    if (node.translation.size() == 3) {
      glTranslated(node.translation[0], node.translation[1],
                   node.translation[2]);
    }
  }

  // std::cout << "node " << node.name << ", Meshes " << node.meshes.size() <<
  // std::endl;

  // std::cout << it->first << std::endl;
  // FIXME(syoyo): Refactor.
  // DrawCurves(scene, it->second);
  if (node.mesh > -1) {
    assert(node.mesh < model.meshes.size());
    DrawMesh(model, model.meshes[node.mesh]);
  }

  // Draw child nodes.
  for (size_t i = 0; i < node.children.size(); i++) {
    assert(node.children[i] < model.nodes.size());
    DrawNode(model, model.nodes[node.children[i]]);
  }

  glPopMatrix();
}

static void DrawModel(tinygltf::Model &model) {
#if 0
	std::map<std::string, tinygltf::Mesh>::const_iterator it(scene.meshes.begin());
	std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(scene.meshes.end());

	for (; it != itEnd; it++) {
		DrawMesh(scene, it->second);
		DrawCurves(scene, it->second);
	}
#else
  // If the glTF asset has at least one scene, and doesn't define a default one
  // just show the first one we can find
  assert(model.scenes.size() > 0);
  int scene_to_display = model.defaultScene > -1 ? model.defaultScene : 0;
  const tinygltf::Scene &scene = model.scenes[scene_to_display];
  for (size_t i = 0; i < scene.nodes.size(); i++) {
    DrawNode(model, model.nodes[scene.nodes[i]]);
  }
#endif
}

static void Init() {
  trackball(curr_quat, 0, 0, 0, 0);

  eye[0] = 0.0f;
  eye[1] = 0.0f;
  eye[2] = CAM_Z;

  lookat[0] = 0.0f;
  lookat[1] = 0.0f;
  lookat[2] = 0.0f;

  up[0] = 0.0f;
  up[1] = 1.0f;
  up[2] = 0.0f;
}

static void PrintNodes(const tinygltf::Scene &scene) {
  for (size_t i = 0; i < scene.nodes.size(); i++) {
    std::cout << "node.name : " << scene.nodes[i] << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "glview input.gltf <scale>" << std::endl;
    std::cout << "defaulting to example cube model" << std::endl;
  }

  float scale = 0.4f;
  if (argc > 2) {
    scale = atof(argv[2]);
  }

  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

#ifdef _WIN32
#ifdef _DEBUG
  std::string input_filename(argv[1] ? argv[1]
                                     : "../../../models/Sponza/glTF/Sponza.gltf");
#endif
#else
  std::string input_filename(argv[1] ? argv[1] : "../../models/Cube/Cube.gltf");
#endif

  std::string ext = GetFilePathExtension(input_filename);

  bool ret = false;
  if (ext.compare("glb") == 0) {
    // assume binary glTF.
    ret =
        loader.LoadBinaryFromFile(&model, &err, &warn, input_filename.c_str());
  } else {
    // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
  }

  if (!warn.empty()) {
    printf("Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    printf("ERR: %s\n", err.c_str());
  }
  if (!ret) {
    printf("Failed to load .glTF : %s\n", argv[1]);
    exit(-1);
  }

  Init();

  // DBG
  PrintNodes(model.scenes[model.defaultScene > -1 ? model.defaultScene : 0]);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW." << std::endl;
    return -1;
  }

  std::stringstream ss;
  ss << "Simple glTF viewer: " << input_filename;

  std::string title = ss.str();

  window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to open GLFW window. " << std::endl;
    glfwTerminate();
    return 1;
  }

  glfwGetWindowSize(window, &width, &height);

  glfwMakeContextCurrent(window);

  // Callback
  glfwSetWindowSizeCallback(window, reshapeFunc);
  glfwSetKeyCallback(window, keyboardFunc);
  glfwSetMouseButtonCallback(window, clickFunc);
  glfwSetCursorPosCallback(window, motionFunc);

  glewExperimental = true;  // This may be only true for linux environment.
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW." << std::endl;
    return -1;
  }

  reshapeFunc(window, width, height);

  GLuint vertId = 0, fragId = 0, progId = 0;

#ifdef _WIN32
#ifdef _DEBUG
  const char *shader_frag_filename = "../shader.frag";
  const char *shader_vert_filename = "../shader.vert";
#endif
#else
  const char *shader_frag_filename = "shader.frag";
  const char *shader_vert_filename = "shader.vert";
#endif

  if (false == LoadShader(GL_VERTEX_SHADER, vertId, shader_vert_filename)) {
    return -1;
  }
  CheckErrors("load vert shader");

  if (false == LoadShader(GL_FRAGMENT_SHADER, fragId, shader_frag_filename)) {
    return -1;
  }
  CheckErrors("load frag shader");

  if (false == LinkShader(progId, vertId, fragId)) {
    return -1;
  }

  CheckErrors("link");

  {
    // At least `in_vertex` should be used in the shader.
    GLint vtxLoc = glGetAttribLocation(progId, "in_vertex");
    if (vtxLoc < 0) {
      printf("vertex loc not found.\n");
      exit(-1);
    }
  }

  glUseProgram(progId);
  CheckErrors("useProgram");

  SetupMeshState(model, progId);
  // SetupCurvesState(model, progId);
  CheckErrors("SetupGLState");

  std::cout << "# of meshes = " << model.meshes.size() << std::endl;

  while (glfwWindowShouldClose(window) == GL_FALSE) {
    glfwPollEvents();
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);

    GLfloat mat[4][4];
    build_rotmatrix(mat, curr_quat);

    // camera(define it in projection matrix)
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluLookAt(eye[0], eye[1], eye[2], lookat[0], lookat[1], lookat[2], up[0],
              up[1], up[2]);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMultMatrixf(&mat[0][0]);

    glScalef(scale, scale, scale);
    updateAnimation(0.03,progId);
    DrawModel(model);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glFlush();

    glfwSwapBuffers(window);
  }

  glfwTerminate();
}
