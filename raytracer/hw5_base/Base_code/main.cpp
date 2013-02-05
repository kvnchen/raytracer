#include "stdafx.h"
#include <stack>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <string>
#include "Transform.h"
//#include <GL/glew.h>
//#include <GL/glut.h>

using namespace std ; 

vec3 center; 
vec3 eye; // The (regularly updated) vector coordinates of the eye location 
vec3 up;  // The (regularly updated) vector coordinates of the up location 

const float ip_corners[4][3] = { {-1, 1, -1}, {1, 1, -1}, {1, -1, -1}, {-1, -1, -1}};

int width, height;  // assigned from size later
const int maxwidth = 500;
const int maxheight = 500;
int depth = 3; // maximum recursive depth
float depth_of_field = 0;
//enum shape {sphere, triangle} ; // geometry
int size[2] ;   // size of the image
//vec3 lights [];
int numlights = 0;
float attenuation[3] = {1, 0, 0};

string filename;     // if a filename is specified, this will be it
float camera[10];  // 0-2: lookfrom, 3-5: lookat, 6-8: up 9: fov

vec3 vertarray[1000]; // array storing every single vertex, numbered 0 through 999
int facearray[2000][3]; // array storing faces (triangles, up to 2000), which consist of 3 verticies each
int maxverts; // just a convenience variable
int maxvertnorms; // max num of verticies w/ normals
float vertexnormalarray[1000][6]; // x, y, z, nx, ny, nz
int vertexcounter = 0;
int trianglecounter = 0;
const int maxobjects = 50;

vec3 g_ambient;
vec3 g_diffuse;
vec3 g_specular;
vec3 g_emission; 
float g_shininess ;
float refraction;

vec3 current_translate;
vec4 current_rotate;
vec3 current_scale;

const int directional = 0;
const int point = 1;

float shadowoffset = 0.2;

// Experiment: perlin noise

struct perlin
{
  int p[512];
  perlin(void);
  static perlin & getInstance(){static perlin instance; return
instance;}
};

static int
permutation[] = { 151,160,137,91,90,15,
  131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
  190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
  88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
  77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
  102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208,89,18,169,200,196,
  135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
  5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
  23,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,43,172,9,
  129,22,39,253,19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
  251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
  49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127,4,150,254,
  138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

static double fade(double t)
{ 
  return t * t * t * (t * (t * 6 - 15) + 10);
}

static double lerp(double t, double a, double b) { 
  return a + t * (b - a);
}

static double grad(int hash, double x, double y, double z) {
  int h = hash & 15;
  // CONVERT LO 4 BITS OF HASH CODE
  double u = h<8||h==12||h==13 ? x : y, // INTO 12 GRADIENT DIRECTIONS.
  v = h < 4||h == 12||h == 13 ? y : z;
  return ((h & 1) == 0 ? u : -u) + ((h&2) == 0 ? v : -v);
}

double noise(double x, double y, double z) {
  perlin & myPerlin = perlin::getInstance();
  int X = (int)floor(x) & 255, // FIND UNIT CUBE THAT
      Y = (int)floor(y) & 255, // CONTAINS POINT.
      Z = (int)floor(z) & 255;
  x -= floor(x);                   // FIND RELATIVE X,Y,Z
  y -= floor(y);                   // OF POINT IN CUBE.
  z -= floor(z);
  double u = fade(x),              // COMPUTE FADE CURVES
         v = fade(y),              // FOR EACH OF X,Y,Z.
         w = fade(z);
  int A = myPerlin.p[X]+Y,    // HASH COORDINATES OF
      AA = myPerlin.p[A]+Z,   // THE 8 CUBE CORNERS,
      AB = myPerlin.p[A+1]+Z, 
      B = myPerlin.p[X+1]+Y, 
      BA = myPerlin.p[B]+Z, 
      BB = myPerlin.p[B+1]+Z;

  return 
    lerp(w, lerp(v, lerp(u, grad(myPerlin.p[AA], x, y, z),      // AND ADD  
                           grad(myPerlin.p[BA], x-1, y, z)),    // BLENDED
                   lerp(u, grad(myPerlin.p[AB], x, y-1, z),     // RESULTS
                           grad(myPerlin.p[BB], x-1, y-1, z))), // FROM 8
           lerp(v, lerp(u, grad(myPerlin.p[AA+1], x, y, z-1),   // CORNERS
                           grad(myPerlin.p[BA+1], x-1, y, z-1)),// OF CUBE
                   lerp(u, grad(myPerlin.p[AB+1], x, y-1, z-1 ),
                           grad(myPerlin.p[BB+1], x-1, y-1, z-1 ))));
}

perlin::perlin (void) 
{ 
  for (int i=0; i < 256 ; i++) {
    p[256+i] = p[i] = permutation[i];
}
}

float unifRand()
{
    return rand() / float(RAND_MAX);
}


struct Color {
	int r, g, b;
};

class Light {
public:
	bool type; // 0 directional, 1 point
	vec3 pos; // Either the position for a point light, or direction for a directional
	//Color color;
	vec3 intensity;
	// attenuation too!
}lights[50];
int lightcounter = 0;

class Ray {
	public:
		vec4 pos;
		vec4 dir;
		bool is_shadowray;
		float t_min, t_max;   // these might be necessary for intersection calculations
		void RayThruPixel(int,int); // and a camera too
		float current_index; // initially 1
};

void Ray::RayThruPixel(int i, int j) {
	// i,j are position of pixel, i -> height, w -> width
	// for vec4's, w term: 0 indicates vector (direction), 1 indicates point
	is_shadowray = 0;
	current_index = 1;

	vec3 a = eye - center;
	vec3 b = up;
	vec3 w = glm::normalize(a);
	vec3 u = glm::normalize(glm::cross(b, w)); 
	vec3 v = glm::cross(w, u);

	// Handling proper aspect ratio
	float aspect = ((float) width / (float) height);
	float fovy = camera[9];
	float fovx = fovy;
	// Lecture implementation, radians conversion
	fovy = fovy *(pi/180);
	fovx = fovx *(pi/180) ;

	//For depth of field, we randomly nudge i & j by a random amount, and redefine the focal plane

	float alpha = tan(aspect*fovx/2)*((float) (j - (width/2)) / (width/2));
	float beta = tan(fovy/2)*( (float) ( (height/2) - i) / (height/2)); 
	pos = vec4(eye,1);  // now they are set as vec4's
	//vec3 temp = glm::normalize(alpha*u + beta*v - w);

	if (depth_of_field >= 0) {
		vec3 rnd(( ((float) (rand() % 10)) / 50), ( ((float) (rand() % 10)) / 50), 0);
		vec3 neye = eye + rnd; // new eye in 3D
		vec4 neweye = pos + vec4(rnd, 0);  // new eye in 4D

		vec3 temp = alpha*u + beta*v - w;
		dir = glm::normalize(vec4(temp,0));
		float D = 1; // eye to image plane (always 1)
		float d = sqrt(pow(temp[0], 2) + pow(temp[1], 2) + pow(temp[2], 2));  // dist of eye to pixel
		// image plane is 1 away from eye 		
		
		vec4 p = pos + (d/ (D / (D + 4.5)) )*dir; // the intersection point of the ray with the focal point
		vec4 newdir = glm::normalize(p - neweye);
		pos = neweye;
		dir = newdir;
	}
	else {
		vec3 temp = alpha*u + beta*v - w;
		dir = glm::normalize(vec4(temp,0));
	}
}

// Shape is a basic abstract class, holds intersect method
class Shape {
public:
	Color color;    // RGB color, from scale of 0 to 1
	float intersect(Ray); 
	vec3 ambient ;  // Lighting properties, as vec3's
	vec3 diffuse ; 
	vec3 specular ;
	vec3 emission ; 
	float shininess ; // simple float
	mat4 matrix ;  // Transformation matrix, mat4
	vec4 normal; // to get around messy transformation issues
	float refractive_index; // n, index of refraction
};

class Sphere: public Shape {
	
public:
	float args[4];  //float x, y, z, radius;
	float intersect(Ray);
}spheres[100];
int spherecounter = 0;    // Tracks how many spheres in scene

float Sphere::intersect(Ray r) {
	// actual implementation goes here
	float ret = 999999;
	vec4 pos = r.pos;
	vec4 dir = r.dir;
	//if (r.is_shadowray == true) {
	//	r.t_min = 0.001; // arbitrary?
	//}
	r.t_min = 0;
	// In the case of a miss, r.t_min is 0, and returned t is 999999 (6 9's)
	vec4 cen = vec4(args[0], args[1], args[2],1); //center of the sphere
	float A = glm::dot(dir,dir);
	float B = 2*glm::dot(dir, pos - cen);
	float C = glm::dot(pos-cen, pos-cen) - args[3]*args[3];
	float det = B*B - 4*A*C;
	if (det >= 0) {
		float root1 = (-B + sqrt(det))/2*A;
		float root2 = (-B - sqrt(det))/2*A;
		float min;
		if (root1 < root2) {
			min = root1;
		} else {
			min = root2;
		}
		if (det = 0) {
			ret = root1;
		} else if (root1 >= 0 && root2 >= 0) {
			ret = min;
		} else if (root1 >=0 && root2 < 0) {
			ret = root1;
		} else if (root1 < 0 && root2 >= 0) {
			ret = root2;
		}
	}
	//if (ret >= r.t_min) {
		return ret;
	//}
	//else {
	//	return 999999;
	//}
}


//bool Sphere::intersect(Ray r) {
	// actually implementation goes here
//	return 0;
//}

class Triangle: public Shape {
public:
	vec3 vertexarray[3];  // 3 verticies of a triangle
	float intersect(Ray);
}triangles[500];

float Triangle::intersect(Ray r) {
	// actual implementation goes here
	float ret = 999999;
	vec4 pos = r.pos;
	vec4 dir = r.dir;
	r.t_min = 0;
	vec3 vert_a = vertexarray[0];
	vec3 vert_b = vertexarray[1];
	vec3 vert_c = vertexarray[2];
	normal = normal = glm::normalize(vec4(glm::normalize(glm::cross(vert_c - vert_a,vert_b - vert_a)),0));
	float a = vert_a[0] - vert_b[0]; // Xa - Xb
	float b = vert_a[1] - vert_b[1]; // Ya - Yb
	float c = vert_a[2] - vert_b[2]; // Za - Zb
	float d = vert_a[0] - vert_c[0];
	float e = vert_a[1] - vert_c[1];
	float f = vert_a[2] - vert_c[2];
	float g = dir[0];
	float h = dir[1];
	float i = dir[2];
	float j =  vert_a[0] - pos[0];
	float k =  vert_a[1] - pos[1];
	float l =  vert_a[2] - pos[2];
	float beta;
	float gamma;
	float t;
	float M = a*(e*i-h*f)+b*(g*f-d*i)+c*(d*h-e*g);
	beta = (j*(e*i-h*f)+k*(g*f-d*i)+l*(d*h-e*g))/M;
	gamma = (i*(a*k-j*b)+h*(j*c-a*l)+g*(b*l-k*c))/M;
	t = -(f*(a*k-j*b)+e*(j*c-a*l)+d*(b*l-k*c))/M;
	vec4 n = vec4(glm::normalize(glm::cross(vertexarray[2]-vertexarray[0],vertexarray[1]-vertexarray[0])),0);
	if (glm::dot(dir,n) != 0 && (t >= r.t_min) && (gamma >= 0) && (gamma <= 1) && (beta >= 0) && (beta <= 1 - gamma)) {
		ret = t;
	}
	return ret;
}

class Image {      // stores the color values and has method writeIamge?
public:
	Color ** colors;
	void writeImage();
	void initialize();
};

void Image::initialize() {  // magic memory allocation bullshit
	int size_x = height;
	int size_y = width;
	colors = (Color**) malloc (size_x * sizeof(Color *));
	for (int i = 0; i < size_x; i++) {
		colors[i] = (Color*) malloc(size_y * sizeof(Color));
	}
	//colors = (Color*) malloc (10000);
}

void Image::writeImage() {
	ofstream myfile; //("output.ppm"); //say we had an output file
	myfile.open ("output.ppm");
	if (myfile.is_open()) {
		myfile << "P3 \n";
		myfile << width << " " << height << " \n";
		myfile << "255 \n";
		//for (int i = height - 1; i >= 0; i--) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {   // order is changed since ppm starts from top left corner
				myfile << colors[i][j].r << " " << colors[i][j].g << " " << colors[i][j].b << " ";
				//myfile << 0 << " " << 0 << " " << 0 << " ";
			}
			myfile << "\n";
		}
		myfile.close();
	} 
}

// **************************** Raytracer stuff

class Intersection {
	public:
		Shape s;
		Ray r;
		vec4 normal;
		Intersection (Ray);
		bool flag;
		vec4 point; // The intersection point, for use in transformations
};

Intersection::Intersection(Ray ray) {
	flag = false;
	float min = 999999;
	//r.t_min = min;
	for (int i = 0; i < spherecounter; i++) {
		Sphere sphere = spheres[i];
		
		Ray sray;
		sray.pos = ray.pos*glm::transpose(glm::inverse(sphere.matrix));
		vec4 tempdir = ray.dir*glm::transpose(glm::inverse(sphere.matrix));
		sray.dir = glm::normalize(tempdir);
		float t = sphere.Sphere::intersect(sray); //we now have the transformed intersection "t"
		vec4 newpoint = sray.pos + t*sray.dir;
		float temp = t;   // temp holds the object-space t value
		t = t/sqrt(glm::dot(tempdir, tempdir)); // returns a proper t in world coordinates. It seems to make everything really slow...
		
		//float t = sphere.Sphere::intersect(ray);
		if (t < min && temp < 999999) { // the min has to be world coordinate min, so here the problem comes in
			min = t; // now are we returning the correct t?
			point = newpoint*glm::transpose(sphere.matrix); // intersection point p turns into Mp
			s = spheres[i];
			flag = true;
			r = sray;
			r.current_index = ray.current_index;
			r.dir = sray.dir*glm::transpose(sphere.matrix);
			r.t_min = min;
			normal = glm::normalize(glm::normalize( (sray.pos + sray.dir*temp) - vec4(sphere.args[0], sphere.args[1], sphere.args[2],1))*glm::inverse(sphere.matrix));   // Normals are hit with inverse transpose
			//normal = glm::normalize(((ray.pos + ray.dir*t) - vec4(sphere.args[0], sphere.args[1], sphere.args[2],1)));
		}
	}
	for (int i = 0; i < trianglecounter; i++) {	
		Triangle triangle = triangles[i];
		Ray tray;
		
		tray.pos = ray.pos*glm::transpose(glm::inverse(triangle.matrix));
		vec4 tempdir = ray.dir*glm::transpose(glm::inverse(triangle.matrix));
		tray.dir = glm::normalize(tempdir);
		float t = triangle.Triangle::intersect(tray);
		float temp = t;
		vec4 newpoint = tray.pos + t*tray.dir;
		t = t/sqrt(glm::dot(tempdir, tempdir));
		
		//float t = triangle.Triangle::intersect(ray);
		if (t < min && temp < 999999) {
			min = t;
			point = newpoint*glm::transpose(triangle.matrix); // intersection point p turns into Mp
			s = triangles[i];
			flag = true;
			//r = ray;
			r = tray;
			r.current_index = ray.current_index;
			r.dir = tray.dir*glm::transpose(triangle.matrix);
			r.t_min = min;
			normal = -glm::normalize(triangle.normal*glm::inverse(triangle.matrix));
			//vec4 A = vec4(triangle.vertexarray[0], 1)*glm::transpose(glm::inverse(triangle.matrix));
			//vec4 B = vec4(triangle.vertexarray[1], 1)*glm::transpose(glm::inverse(triangle.matrix));
			//vec4 C = vec4(triangle.vertexarray[2], 1)*glm::transpose(glm::inverse(triangle.matrix));
			//normal = glm::normalize(vec4(glm::cross(C-A,B-A),0));
			// Since cross wont work for vec4s, we calculate normals in the intersect function
			//normal = glm::normalize(glm::normalize(glm::cross(C-A,B-A))*glm::inverse(triangle.matrix) );
		}
	}
	
}

struct Colorf {
	float r, g, b;
};

class Shading {
public: 
	Colorf shadecolor;
	void shade(Intersection, int); // also must take in all lights later
};


// shading code


/*
void Shading::shade(Intersection hit, int d) {
	if (d < depth && (hit.flag == true)) { //recursive depth 
		Shape shape = hit.s;
		if (hit.r.current_index != 1) {
			hit.normal = -hit.normal;
		}
		vec3 veccolor = vec3(shape.ambient[0] + shape.emission[0], shape.ambient[1] + shape.emission[1], 
			shape.ambient[2] + shape.emission[2]);
		for (int i = 0; i < lightcounter; i++) {
			Ray shadow; // generate a shadow ray
			//shadow.pos = (hit.r.pos + hit.r.dir*hit.r.t_min + hit.normal*0.001);
			shadow.pos = (hit.point + hit.normal*0.001);  // now, replace calculating point with hit.point

			vec4 lightposition;
			vec4 lightdirection;
			if (lights[i].type != 0) {  // point light
				lightposition = vec4(lights[i].pos,1);
				lightdirection = glm::normalize(lightposition - shadow.pos);
				//attenuation[0] = 0;
				//attenuation[1] = 0;
				//attenuation[2] = 1;
			} else {
				lightposition = vec4(lights[i].pos,0); // the 3d position in space
				lightdirection = glm::normalize(lightposition);
				//attenuation[0] = 1;
				//attenuation[1] = 0;
				//attenuation[2] = 0;
			}
			float r = sqrt(glm::dot(lightposition -shadow.pos,lightposition - shadow.pos));
			vec3 atten = lights[i].intensity/(attenuation[0] + attenuation[1]*r + attenuation[2]*pow(r,2)); 
			vec4 eyedirn = glm::normalize(vec4(eye,1) - shadow.pos);
			vec4 half = glm::normalize(lightdirection + eyedirn);

			if (lights[i].type == 1) {  // point light
				shadow.dir = glm::normalize(vec4(lights[i].pos,1) - shadow.pos); // this is correct
			}
			else {
				shadow.dir = glm::normalize(vec4(lights[i].pos,1));
			} 

			shadow.is_shadowray = 1;
			Intersection in (shadow);
			float v;
			if (in.flag == true && d == 0) {
				v = 0;
			}
			else { 
				v = 1;
			}
			
			veccolor[0] = veccolor[0] + v*atten[0]*(shape.diffuse[0]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[0]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			veccolor[1] = veccolor[1] + v*atten[1]*(shape.diffuse[1]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[1]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			veccolor[2] = veccolor[2] + v*atten[2]*(shape.diffuse[2]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[2]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );

			//veccolor[0] = veccolor[0] + shape.specular[0]*reflectshade.shadecolor.r;
			//veccolor[1] = veccolor[1] + shape.specular[1]*reflectshade.shadecolor.g;
			//veccolor[2] = veccolor[2] + shape.specular[2]*reflectshade.shadecolor.b;
			// Color Clamping
			
		}

		float n1 = hit.r.current_index; // and 0 depth, this is 1
		float n2 = shape.refractive_index; // based on material properties
		if (n1 == n2) {
			n2 = 1;
		}
		float r0 = pow((n2 - 1)/(n2 + 1), 2);
		float R  = r0 + (1 - r0)*pow((1 - glm::dot(-hit.r.dir, hit.normal)), 5);


		// Reflection rays, sent after all other rays are cast. Does this make sense?
		if (hit.r.current_index == 1) {
		Ray rray;
			
		rray.dir = glm::normalize(hit.r.dir - (2*glm::dot(hit.r.dir,hit.normal))*hit.normal);
		rray.pos = hit.point + rray.dir*0.001;
		Intersection reflect(rray);
		Shading reflectshade;
		reflectshade.shade(reflect,d+1);

		veccolor[0] = veccolor[0] + shape.specular[0]*((float) reflectshade.shadecolor.r);
		veccolor[1] = veccolor[1] + shape.specular[1]*((float) reflectshade.shadecolor.g);
		veccolor[2] = veccolor[2] + shape.specular[2]*((float) reflectshade.shadecolor.b);
		}

		if (shape.refractive_index > 0) { // checks if material is transparent
		Ray refract_ray;

		float n3 = (n1 / n2);
		float c2 = 1 - (float) pow(n3, 2)*(1 - (float)pow(glm::dot(hit.r.dir, hit.normal), 2) );

		if (c2 > 0) {
		refract_ray.dir = glm::normalize(n3*(hit.r.dir - hit.normal*glm::dot(hit.r.dir, hit.normal)) - hit.normal*( (float) sqrt(c2) ) );
		refract_ray.pos = hit.point + refract_ray.dir*0.001;
		refract_ray.current_index = n2;
		Intersection refract(refract_ray);
		Shading refractshade;
		refractshade.shade(refract,d+1);

		veccolor[0] = veccolor[0] + ((float) refractshade.shadecolor.r); // Transmission is set to 1 atm
		veccolor[1] = veccolor[1] + ((float) refractshade.shadecolor.g);
		veccolor[2] = veccolor[2] + ((float) refractshade.shadecolor.b);

		}
		}

		// Photo exposure, a type of mapping of values. Still clip a lower bound
		
		//float exposure = -2.00;
		
		//veccolor[0] = 1 - (float)exp(veccolor[0]*exposure);
		//veccolor[1] = 1 - (float)exp(veccolor[1]*exposure);
		//veccolor[2] = 1 - (float)exp(veccolor[2]*exposure);
		
		if (veccolor[0] > 1) {
				veccolor[0] = 1;
		}
		if (veccolor[0] < 0) {
				veccolor[0] = 0;
		}
		if (veccolor[1] > 1) {
				veccolor[1] = 1;
		}
		if (veccolor[1] < 0) {
			veccolor[1] = 0;
		}
		if (veccolor[2] > 1) {
				veccolor[2] = 1;
		}
		if (veccolor[2] < 0) {
				veccolor[2] = 0;
		}
		shadecolor.r =  veccolor[0];
		shadecolor.g =  veccolor[1];
		shadecolor.b =  veccolor[2];
	}


	else {
		shadecolor.r =  0;
		shadecolor.g =  0;
		shadecolor.b =  0;
	}
}
*/
	
void Shading::shade(Intersection hit, int d) {
	if (d < depth && (hit.flag == true)) { //recursive depth 
	Shape shape = hit.s;
	if (hit.r.current_index != 1) {
			hit.normal = -hit.normal;
		}
		vec3 veccolor = vec3(shape.ambient[0] + shape.emission[0], shape.ambient[1] + shape.emission[1], 
			shape.ambient[2] + shape.emission[2]);
		for (int i = 0; i < lightcounter; i++) {
			Ray shadow; // generate a shadow ray
			//shadow.pos = (hit.r.pos + hit.r.dir*hit.r.t_min + hit.normal*0.001);
			shadow.pos = (hit.point + hit.normal*0.001);  // now, replace calculating point with hit.point

			vec4 lightposition;
			vec4 lightdirection;
			vec4 lightside;
			vec4 lightup;
			if (lights[i].type != 0) {  // point light
				vec4 lightpositiontest = vec4(lights[i].pos,1);
				vec4 lightdirectiontest = glm::normalize(lightpositiontest - shadow.pos);
				
				lightside = glm::normalize(vec4(glm::cross(vec3(lightdirectiontest), up),0)); //might not work for lights directly above or below a point
				lightup = glm::normalize(vec4(glm::cross(vec3(lightside),vec3(lightdirectiontest)),0)); //fixing lightup will fix this as well
				lightposition = lightpositiontest + unifRand()*lightup*shadowoffset + unifRand()*lightside*shadowoffset;  // Randomized shadows
				//lightposition = lightpositiontest;
				lightdirection = glm::normalize(lightposition - shadow.pos); 
				
				//attenuation[0] = 0;
				//attenuation[1] = 0;
				//attenuation[2] = 1;
			} else {
				lightposition = vec4(lights[i].pos,1); // the 3d position in space
				lightdirection = glm::normalize(vec4(vec3(lightposition),0));
				//attenuation[0] = 1;
				//attenuation[1] = 0;
				//attenuation[2] = 0;
			}
			/*
			if (lights[i].type != 0) {
				for (int y = -1; y < 2; y++) {
					for (int x = -1; x < 2; x++) {
						vec4 lightpositionhelper = lightposition + x*lightside*shadowoffset + y*lightup*shadowoffset;
						vec4 lightdirectionhelper = glm::normalize(lightpositionhelper - shadow.pos); 

						float r = sqrt(glm::dot(lightpositionhelper -shadow.pos,lightpositionhelper - shadow.pos));
						vec3 atten = lights[i].intensity/(attenuation[0] + attenuation[1]*r + attenuation[2]*pow(r,2));
						vec4 eyedirn = glm::normalize(vec4(eye,1) - shadow.pos);
						vec4 half = glm::normalize(lightdirectionhelper + eyedirn);

						shadow.dir = lightdirectionhelper;

						shadow.is_shadowray = 1;
						Intersection in (shadow);
						float v;
						if (in.flag == true && d == 0) {
							v = 0;
						}
						else { 
							v = 1;
						}
			
						veccolor[0] = veccolor[0] + (1.0/9.0)*v*atten[0]*(shape.diffuse[0]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[0]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
						veccolor[1] = veccolor[1] + (1.0/9.0)*v*atten[1]*(shape.diffuse[1]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[1]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
						veccolor[2] = veccolor[2] + (1.0/9.0)*v*atten[2]*(shape.diffuse[2]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[2]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
					}
				}
			} else {
				float r = sqrt(glm::dot(lightposition -shadow.pos,lightposition - shadow.pos));
				vec3 atten = lights[i].intensity/(attenuation[0] + attenuation[1]*r + attenuation[2]*pow(r,2));
				vec4 eyedirn = glm::normalize(vec4(eye,1) - shadow.pos);
				vec4 half = glm::normalize(lightdirection + eyedirn);

				shadow.dir = lightdirection;

				shadow.is_shadowray = 1;
				Intersection in (shadow);
				float v;
				if (in.flag == true && d == 0) {
					v = 0;
				}
				else { 
					v = 1;
				}
			
				veccolor[0] = veccolor[0] + v*atten[0]*(shape.diffuse[0]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[0]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
				veccolor[1] = veccolor[1] + v*atten[1]*(shape.diffuse[1]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[1]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
				veccolor[2] = veccolor[2] + v*atten[2]*(shape.diffuse[2]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[2]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			}
			*/
			
			
			float r = sqrt(glm::dot(lightposition -shadow.pos,lightposition - shadow.pos));
			vec3 atten = lights[i].intensity/(attenuation[0] + attenuation[1]*r + attenuation[2]*pow(r,2));
			vec4 eyedirn = glm::normalize(vec4(eye,1) - shadow.pos);
			vec4 half = glm::normalize(lightdirection + eyedirn);

			shadow.dir = lightdirection;
			
			/*
			if (lights[i].type == 1) {  // point light
				shadow.dir = glm::normalize(vec4(lights[i].pos,1) - shadow.pos); // this is correct
			}
			else {
				shadow.dir = glm::normalize(vec4(lights[i].pos,0));
			} 
			*/
			
			shadow.is_shadowray = 1;
			Intersection in (shadow);
			float v;
			if (in.flag == true && d == 0) {
				v = 0;
			}
			else { 
				v = 1;
			}
			
			veccolor[0] = veccolor[0] + v*atten[0]*(shape.diffuse[0]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[0]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			veccolor[1] = veccolor[1] + v*atten[1]*(shape.diffuse[1]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[1]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			veccolor[2] = veccolor[2] + v*atten[2]*(shape.diffuse[2]*max((float) glm::dot(hit.normal,lightdirection), (float) 0) + shape.specular[2]*pow((float) max((float) glm::dot(hit.normal,half), (float) 0), shape.shininess) );
			
			//veccolor[0] = veccolor[0] + shape.specular[0]*reflectshade.shadecolor.r;
			//veccolor[1] = veccolor[1] + shape.specular[1]*reflectshade.shadecolor.g;
			//veccolor[2] = veccolor[2] + shape.specular[2]*reflectshade.shadecolor.b;
			// Color Clamping
			
			
		}
		float n1 = hit.r.current_index; // and 0 depth, this is 1
		float n2 = shape.refractive_index; // based on material properties
		if (n1 == n2) {
			n2 = 1;
		}
		float r0 = pow((n2 - 1)/(n2 + 1), 2);
		float R  = r0 + (1 - r0)*pow((1 - glm::dot(-hit.r.dir, hit.normal)), 5);


		// Reflection rays, sent after all other rays are cast. Does this make sense?
		if (hit.r.current_index == 1) {
		Ray rray;
			
		rray.dir = glm::normalize(hit.r.dir - (2*glm::dot(hit.r.dir,hit.normal))*hit.normal);
		rray.pos = hit.point + rray.dir*0.001;
		Intersection reflect(rray);
		Shading reflectshade;
		reflectshade.shade(reflect,d+1);

		veccolor[0] = veccolor[0] + shape.specular[0]*((float) reflectshade.shadecolor.r);
		veccolor[1] = veccolor[1] + shape.specular[1]*((float) reflectshade.shadecolor.g);
		veccolor[2] = veccolor[2] + shape.specular[2]*((float) reflectshade.shadecolor.b);
		}

		if (shape.refractive_index > 0) { // checks if material is transparent
		Ray refract_ray;

		float n3 = (n1 / n2);
		float c2 = 1 - (float) pow(n3, 2)*(1 - (float)pow(glm::dot(hit.r.dir, hit.normal), 2) );

		if (c2 > 0) {
		refract_ray.dir = glm::normalize(n3*(hit.r.dir - hit.normal*glm::dot(hit.r.dir, hit.normal)) - hit.normal*( (float) sqrt(c2) ) );
		refract_ray.pos = hit.point + refract_ray.dir*0.001;
		refract_ray.current_index = n2;
		Intersection refract(refract_ray);
		Shading refractshade;
		refractshade.shade(refract,d+1);

		veccolor[0] = veccolor[0] + ((float) refractshade.shadecolor.r); // Transmission is set to 1 atm
		veccolor[1] = veccolor[1] + ((float) refractshade.shadecolor.g);
		veccolor[2] = veccolor[2] + ((float) refractshade.shadecolor.b);

		}
		}


		if (veccolor[0] > 1) {
				veccolor[0] = 1;
		}
		if (veccolor[0] < 0) {
				veccolor[0] = 0;
		}
		if (veccolor[1] > 1) {
				veccolor[1] = 1;
		}
		if (veccolor[1] < 0) {
			veccolor[1] = 0;
		}
		if (veccolor[2] > 1) {
				veccolor[2] = 1;
		}
		if (veccolor[2] < 0) {
				veccolor[2] = 0;
		}
		shadecolor.r =  veccolor[0];
		shadecolor.g =  veccolor[1];
		shadecolor.b =  veccolor[2];
	}


	else {
		shadecolor.r =  0;
		shadecolor.g =  0;
		shadecolor.b =  0;
	}
}





// Other code for HW5, new stuff

// The complete pseudocode:
/*

Image Raytrace (Camera cam, Scene scene, int width, int height) {
	Image image = new Image (width, height) ;
	for (int i = 0; i < height; i++) {                   // These two lines
		for (int j = 0; j < width; j++) {				// are enough to scan the screen	
			Ray ray = RayThroughPixel(cam, i, j);		
			Intersection hit = Intersect(ray, scene);   
			image[i][j] = FindColor(hit);  // a 2D array of color for each pixel, pretty simple
		}
	}
	return image; // 
}


Outputting (C code, for a TGA file):

imageFile.put(min(blue*255.0f,255.0f)).put(min(green*255.0f, 255.0f)).put(min(red*255.0f, 255.0f));

Notes: this has a clamping of floating point precision to 0-255 integer preciscion due to TGA limitations

Classes to implement: 

Scene - presumably holds all geometry and lighting information
Ray - the mathmematical representation of a ray, origin + t*dir
Intersect - something that would help you find color 

*/

void Parser (const char * filename) {
	stack <mat4> transfstack ; 
	transfstack.push(mat4(1.0)) ; //sets initial value to identity
  string str, ret = "" ; 
  ifstream in ; 
  in.open(filename) ; 
  if (in.is_open()) {
    getline (in, str) ; 
	int n = 0;
    while (in) { 
		if ((str.find_first_not_of("\t\r\n") != string::npos) && (str[0] != '#')) {
			string cmd;
			stringstream s(str);
			s >> cmd;
				if (cmd == "directional" && lightcounter < 50) {   // change later
					lights[lightcounter].type = 0; // signifies directional 
					for (int i = 0; i < 6; i++) {  // xyz rgb
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							float x;
							s >> x;
							if (i < 3) {
								lights[lightcounter].pos[i] = x;
							} else {
								lights[lightcounter].intensity[i-3] = x;
							}
						}
					}
					lightcounter++;
				}
				else if (cmd == "point" && lightcounter < 50) {   // change later
					//Light l = lights[lightcounter];
					lights[lightcounter].type = 1; // signifies directional 
					for (int i = 0; i < 6; i++) {  // xyz rgb
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							float x;
							s >> x;
							if (i < 3) {
								lights[lightcounter].pos[i] = x;
							}
							else {
								lights[lightcounter].intensity[i-3] = x;
							}
						}
					}
					lightcounter++;
				}
				else if (cmd == "size") {   // refers to size of image, width and height
					for (int i = 0; i < 2; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> size[i];
						}
					}
				}
				else if (cmd == "attenuation") {   // refers to size of image, width and height
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> attenuation[i];
						}
					}
				}
				else if (cmd == "maxdepth") {   
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
							s >> depth;
					}
				}
				/*
				else if (cmd == "output") {   
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
						 s >> filename; 
					}
				}
				*/
				else if (cmd == "camera") {
					for (int i = 0; i < 10; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> camera[i] ;
						}
					}
				}
				else if (cmd == "maxverts") {   
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
							s >> maxverts;
					}
				}
				else if (cmd == "maxvertnorms") {   
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
							s >> maxvertnorms;
					}
				}
				else if (cmd == "vertex") {
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> vertarray[vertexcounter][i] ;
						}
					}
					vertexcounter++;
				}
				else if (cmd == "tri") {
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							int x;
							s >> x;
							triangles[trianglecounter].vertexarray[i] = vertarray[x];  //conversion from 1 based indexing
						}
					}
					/*
					triangles[trianglecounter].color.r = 0;
					triangles[trianglecounter].color.g = 0;
					triangles[trianglecounter].color.b = 255;
					*/
					triangles[trianglecounter].ambient = g_ambient;
					triangles[trianglecounter].diffuse = g_diffuse;
					triangles[trianglecounter].emission = g_emission;
					triangles[trianglecounter].specular = g_specular;
					triangles[trianglecounter].shininess = g_shininess;
					triangles[trianglecounter].matrix = transfstack.top();
					triangles[trianglecounter].refractive_index = refraction;
					trianglecounter++;
					
				}
				else if (cmd == "sphere") {          // temporarily make sphere red
					for (int i = 0; i < 4; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> spheres[spherecounter].args[i] ;
						}
					}

					/*
					spheres[spherecounter].color.r = 255;
					spheres[spherecounter].color.g = 0;   // remove this stuff later
					spheres[spherecounter].color.b = 0;
					*/
					spheres[spherecounter].ambient = g_ambient;
					spheres[spherecounter].diffuse = g_diffuse;
					spheres[spherecounter].emission = g_emission;
					spheres[spherecounter].specular = g_specular;
					spheres[spherecounter].shininess = g_shininess;
					spheres[spherecounter].matrix = transfstack.top();
					spheres[spherecounter].refractive_index = refraction;
					spherecounter++;
				}
				//          Lighting
				else if (cmd == "ambient") {            // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> g_ambient[i];
						}
					}
				}
				else if (cmd == "diffuse") {          // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> g_diffuse[i] ;
						}
					}
				}
				else if (cmd == "specular") {            // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> g_specular[i] ;
						}
					}
				}
				else if (cmd == "emission") {               // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> g_emission[i] ;
						}
					}
				}
				else if (cmd == "shininess") {            // change later?
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
						s >> g_shininess;
					}
				}
				else if (cmd == "refraction") {            // change later?
					if (s.str().empty()) {
						cerr << "Not enough arguments to 'size'\n";
						throw 2;
					} else {
						s >> refraction;
					}
				}
				else if (cmd == "pushTransform") {
					//flag = true ;
					mat4 top = transfstack.top() ;
					transfstack.push(top) ;
				}
				else if (cmd == "popTransform") {
					//flag = false ;
					
					transfstack.pop() ;
				}
				
				else if (cmd == "translate") {               // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> current_translate[i] ;
						}
					}
					mat4 M = Transform::translate(current_translate[0], current_translate[1], current_translate[2]) ;
					mat4 & T = transfstack.top() ;
					M = glm::transpose(M) ;
					T = T * M ;
				}
				else if (cmd == "rotate") {               // change later?
					for (int i = 0; i < 4; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> current_rotate[i] ;
						}
					}
					mat4 M (0) ;
					vec3 axis(current_rotate[0], current_rotate[1], current_rotate[2]) ;
					mat3 R = Transform::rotate(current_rotate[3], axis) ;

					//Setting up M
					
					M[0][0] = R[0][0] ;
					M[0][1] = R[0][1] ;
					M[0][2] = R[0][2] ;
					M[1][0] = R[1][0] ;
					M[1][1] = R[1][1] ;
					M[1][2] = R[1][2] ;
					M[2][0] = R[2][0] ;
					M[2][1] = R[2][1] ;
					M[2][2] = R[2][2] ;	
					M[0][3] = 0 ;
					M[1][3] = 0 ;
					M[2][3] = 0 ;
					M[3][0] = 0 ;
					M[3][1] = 0 ;
					M[3][2] = 0 ;
					M[3][3] = 1 ;

					mat4 & T = transfstack.top() ;
					M = glm::transpose(M) ;
					T = T * M ;
				}
				else if (cmd == "scale") {               // change later?
					for (int i = 0; i < 3; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> current_scale[i] ;
						}
					}
					mat4 M = Transform::scale(current_scale[0], current_scale[1], current_scale[2]) ;
					
					mat4 & T = transfstack.top() ;
					M = glm::transpose(M) ;
					T = T * M ;
				}
				/*
				else if (cmd == "v") {
					for (int i = 0; i < 4; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							s >> objvertex[vertexcounter][i] ;
						}
					}
					vertexcounter += 1;
				}
				else if (cmd == "f") {
					for (int i = 0; i < 4; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							int x;
							s >> x;
							objface[facecounter][i] = x - 1;  //conversion from 1 based indexing
						}
					}
					facecounter += 1;
				}
				
				else {
					for (int i = 0; i < 9; i++) {
						if (s.str().empty()) {
							cerr << "Not enough arguments to 'size'\n";
							throw 2;
						} else {
							int x;
							s >> tablearray[tablecounter][i];
						}
					}
					tablecounter += 1;
				}
				*/
		}
		getline (in, str) ; 
    }
  }
  else {
    cerr << "Unable to Open File " << filename << "\n" ; 
    throw 2 ; 
  }
}

/*
Image Raytrace (Camera cam, Scene scene, int width, int height) {
	Image image = new Image (width, height) ;
	for (int i = 0; i < height; i++) {                   // These two lines
		for (int j = 0; j < width; j++) {				// are enough to scan the screen	
			Ray ray = RayThroughPixel(cam, i, j);		
			Intersection hit = Intersect(ray, scene);   
			image[i][j] = FindColor(hit);  // a 2D array of color for each pixel, pretty simple
		}
	}
	return image; // 
}
*/

// int argc, char* argv[]

int main () {	
	
	Parser("scene-refraction.test");
	
	//eye = vec3((camera[0] - camera[3]), (camera[1] - camera[4]), (camera[2] - camera[5])) ; 
	eye = vec3(camera[0], camera[1], camera[2]) ;  // lookfrom
	up = vec3(camera[6], camera[7], camera[8]) ; 
	center = vec3(camera[3], camera[4], camera[5]);  // lookat
	width = size[0];
	height = size[1];
	
	
	Image image;
	image.initialize();

	// Depth of field/anti-aliasing considerations

	/*
	for (int i = 0; i < height; i++) {            
		for (int j = 0; j < width; j++) {
			Ray ray;
			ray.RayThruPixel(i, j); // initializes the ray, setting its origin and direction (which is normalized)
			Intersection hit (ray);  // stores the hit object
			if (hit.flag == true) {
				Shading sh;
				sh.shade(hit, 0);
				image.colors[i][j].r = (int) (sh.shadecolor.r*255);
				image.colors[i][j].g = (int) (sh.shadecolor.g*255);
				image.colors[i][j].b = (int) (sh.shadecolor.b*255);
			}
			else {
				image.colors[i][j].r = 0;
				image.colors[i][j].g = 0;
				image.colors[i][j].b = 0;
			}
		}
	}
	image.writeImage();
	return 0;
	*/
	
    for (int i = 0; i < height; i++) {            
        for (int j = 0; j < width; j++) {
            float red = 0;
            float green = 0;
            float blue = 0;
            
            for (float y = i; y < i + 1.0; y += 0.5) {
                for (float x = j; x < j + 1.0; x += 0.5) {
                    Ray ray;
                    ray.RayThruPixel(y, x); // initializes the ray
                    Intersection hit (ray);  // stores the hit object
                    if (hit.flag == true) {
                        Shading sh;
                        sh.shade(hit,0);
                        red += 0.25*sh.shadecolor.r*255;
                        green += 0.25*sh.shadecolor.g*255;
                        blue += 0.25*sh.shadecolor.b*255;
                    }
                }
            }
            
            image.colors[i][j].r = (int) red;
            image.colors[i][j].g = (int) green;
            image.colors[i][j].b = (int) blue;
        }
    }
	image.writeImage();
	return 0;
	
}