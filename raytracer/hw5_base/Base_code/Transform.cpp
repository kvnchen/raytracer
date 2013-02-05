// Transform.cpp: implementation of the Transform class.

#include "stdafx.h"
#include "Transform.h"

//Takes as input the current eye position, and the current up vector.
//up is always normalized to a length of 1.
//eye has a length indicating the distance from the viewer to the origin

// Helper rotation function.  Please implement this.  

mat3 Transform::rotate(const float degrees, const vec3& axis) {
  mat3 R ; 
  // FILL IN YOUR CODE HERE
  // Note: Creates a rotation matrix by the axis-angle formula. Nothing too tricky
  float pi = 3.141592653 ;
  float rad = degrees * (pi / 180);
  mat3 I(1.0);
  mat3 a_t ;
  // Setting up aa_t
  a_t[0][0] = axis[0] * axis[0] ;
  a_t[0][1] = axis[0] * axis[1] ;
  a_t[0][2] = axis[0] * axis[2] ;
  a_t[1][0] = axis[1] * axis[0] ;
  a_t[1][1] = axis[1] * axis[1] ;
  a_t[1][2] = axis[1] * axis[2] ;
  a_t[2][0] = axis[2] * axis[0] ;
  a_t[2][1] = axis[2] * axis[1] ;
  a_t[2][2] = axis[2] * axis[2] ;
  // Setting up astar
  mat3 astar ;
  astar[0][0] = 0 ;
  astar[0][1] = -axis[2] ;
  astar[0][2] = axis[1] ;
  astar[1][0] = axis[2] ;
  astar[1][1] = 0 ;
  astar[1][2] = -axis[0] ;
  astar[2][0] = -axis[1] ;
  astar[2][1] = axis[0] ;
  astar[2][2] = 0 ;
  R = cos(rad)*I + (1 - cos(rad))*a_t + sin(rad)*astar ;
  return R ; 
}

void Transform::left(float degrees, vec3& eye, vec3& up) {

	//FILL IN YOUR CODE HERE
	//rotate around upvector

	mat3 M = Transform::rotate(degrees, glm::normalize(up)) ;
	eye = eye * M ;
}

void Transform::up(float degrees, vec3& eye, vec3& up) {

	//FILL IN YOUR CODE HERE
	//vec3 origin(0, 0, 0) ;
	//a = eye - origin ;
	
	vec3 c = glm::cross(eye, glm::normalize(up)) ;
	// Normalizing
	double cmag = sqrt(pow(c[0], 2) + pow(c[1], 2) + pow(c[2], 2)) ;
	c[0] = c[0] / cmag ;
	c[1] = c[1] / cmag ;
	c[2] = c[2] / cmag ;
	mat3 M = Transform::rotate(degrees, c) ;
	eye = eye * M ;
	up = up * M ;
	vec3 temp = glm::normalize(up);
	up = temp ;
}

mat4 Transform::lookAt(const vec3& eye, const vec3 &center, const vec3& up) {
    mat4 M(0); 
	//FILL IN YOUR CODE HERE
    //You must return a row-major mat4 M that you create from this routine
	//const vec3& eye, const vec3 &center, const vec3& up
	//vec3 eye, vec3 up

	vec3 a ;
	a[0] = eye[0] - center[0] ;
	a[1] = eye[1] - center[1] ;
	a[2] = eye[2] - center[2] ;


	vec3 w;
	// Normalize
	double wmag = sqrt(pow(eye[0], 2) + pow(eye[1], 2) + pow(eye[2], 2)) ;
	w[0] = a[0] / wmag ;
	w[1] = a[1] / wmag ;
	w[2] = a[2] / wmag ;

	vec3 u = glm::cross(up, w);
	// Normalize
	double umag = sqrt(pow(u[0], 2) + pow(u[1], 2) + pow(u[2], 2)) ;
	u[0] = u[0] / umag ;
	u[1] = u[1] / umag ;
	u[2] = u[2] / umag ;

	vec3 v = glm::cross(w, u) ;

	//Define M
	
	//R matrix
	M[0][0] = u[0] ;
	M[0][1] = u[1] ;
	M[0][2] = u[2] ;

	M[1][0] = v[0] ;
	M[1][1] = v[1] ;
	M[1][2] = v[2] ;

	M[2][0] = w[0] ;
	M[2][1] = w[1] ;
	M[2][2] = w[2] ;

	//Bottom row

	M[3][0] = 0 ;
	M[3][1] = 0 ;
	M[3][2] = 0 ;
	M[3][3] = 1 ;

	//New Translation Part

	//Nasty part

	M[0][3] = -(u[0]*a[0]) - (u[1]*a[1]) - (u[2]*a[2]) ;
	M[1][3] = -(v[0]*a[0]) - (v[1]*a[1]) - (v[2]*a[2]) ;
	M[2][3] = -(w[0]*a[0]) - (w[1]*a[1]) - (w[2]*a[2]) ;

	return M ; 
}

mat4 Transform::scale(const float &sx, const float &sy, const float &sz) {
	mat4 M(0) ;

	M[0][0] = sx ;
	M[1][1] = sy ;
	M[2][2] = sz ;
	M[3][3] = 1 ;

	return M ;
}

mat4 Transform::translate(const float &tx, const float &ty, const float &tz)  {
	mat4 M(0) ;

	//defining I part

	M[0][0] = 1 ;
	M[1][1] = 1 ;
	M[2][2] = 1 ;
	M[3][3] = 1 ;

	//the translating part

	M[0][3] = tx ;
	M[1][3] = ty ;
	M[2][3] = tz ;

	return M ;

}

mat4 Transform::perspective(float fovy, float aspect, float zNear, float zFar) {
	//theta = fovy/2
	// d = cot(theta)

	
	float theta = fovy/2 ;
	float rad = theta * (pi / 180) ;
	//float d = 1/(glm::tan(rad)) ;
	float d = cos(rad)/sin(rad) ;

	float A = -(zNear + zFar)/(zFar - zNear) ;
	float B = -(2*zNear*zFar)/(zFar - zNear) ;

	mat4 M(0) ;

	M[0][0] = d/aspect ;
	M[1][1] = d ;
	M[2][2] = A ;
	M[2][3] = B ;
	M[3][2] = -1 ;

	return M ;

}

Transform::Transform()
{

}

Transform::~Transform()
{

}

// Some notes about using glm functions.
// You are ONLY permitted to use glm::dot glm::cross glm::normalize
// Do not use more advanced glm functions (in particular, directly using 
// glm::lookAt is of course prohibited).  

// You may use overloaded operators for matrix-vector multiplication 
// But BEWARE confusion between opengl (column major) and row major 
// conventions, as well as what glm implements. 
// In particular, vecnew = matrix * vecold may not implement what you think 
// it does.  It treats matrix as column-major, in essence using the transpose.
// We recommend using row-major and vecnew = vecold * matrix 
// Preferrably avoid matrix-matrix multiplication altogether for this hw.  
