# version 120 

// Mine is an old machine.  For version 130 or higher, do 
// in vec4 color ;  
// in vec4 mynormal ; 
// in vec4 myvertex ;
// That is certainly more modern

varying vec4 color ;
varying vec3 mynormal ; 
varying vec4 myvertex ; 

uniform int islight ; // are we lighting. 

uniform int is0;
uniform int is1;
uniform int is2;
uniform int is3;
uniform int is4;
uniform int is5;
uniform int is6;
uniform int is7;
uniform int is8;
uniform int is9;

// Assume light 0 and light 1 are both point lights
// The actual light values are passed from the main OpenGL program. 
// This could of course be fancier.  My goal is to illustrate a simple idea. 

uniform vec4 light0posn ; 
uniform vec4 light0color ; 
uniform vec4 light1posn ; 
uniform vec4 light1color ; 
uniform vec4 light2posn ; 
uniform vec4 light2color ; 
uniform vec4 light3posn ; 
uniform vec4 light3color ; 
uniform vec4 light4posn ; 
uniform vec4 light4color ; 
uniform vec4 light5posn ; 
uniform vec4 light5color ; 
uniform vec4 light6posn ; 
uniform vec4 light6color ; 
uniform vec4 light7posn ; 
uniform vec4 light7color ; 
uniform vec4 light8posn ; 
uniform vec4 light8color ; 
uniform vec4 light9posn ; 
uniform vec4 light9color ; 

// Now, set the material parameters.  These could be varying and/or bound to 
// a buffer.  But for now, I'll just make them uniform.  
// I use ambient, diffuse, specular, shininess as in OpenGL.  
// But, the ambient is just additive and doesn't multiply the lights.  

uniform vec4 ambient ; 
uniform vec4 emission;
uniform vec4 diffuse ; 
uniform vec4 specular ; 
uniform float shininess ; 

vec4 ComputeLight (const in vec3 direction, const in vec4 lightcolor, const in vec3 normal, const in vec3 halfvec, const in vec4 mydiffuse, const in vec4 myspecular, const in float myshininess) {

        float nDotL = dot(normal, direction)  ;         
        vec4 lambert = mydiffuse * lightcolor * max (nDotL, 0.0) ;  

        float nDotH = dot(normal, halfvec) ; 
        vec4 phong = myspecular * lightcolor * pow (max(nDotH, 0.0), myshininess) ; 

        vec4 retval = lambert + phong ; 
        return retval ;            
}       

void main (void) 
{       
    if (islight == 0) gl_FragColor = color ; 
    else { 
        // They eye is always at (0,0,0) looking down -z axis 
        // Also compute current fragment position and direction to eye 

        const vec3 eyepos = vec3(0,0,0) ; 
        vec4 _mypos = gl_ModelViewMatrix * myvertex ; 
        vec3 mypos = _mypos.xyz / _mypos.w ; // Dehomogenize current location 
        vec3 eyedirn = normalize(eyepos - mypos) ; 

        // Compute normal, needed for shading. 
        // Simpler is vec3 normal = normalize(gl_NormalMatrix * mynormal) ; 
        vec3 _normal = (gl_ModelViewMatrixInverseTranspose*vec4(mynormal,0.0)).xyz ; 
        vec3 normal = normalize(_normal) ; 

        // Light 0
		vec4 col0 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is0 != 0) {
			vec3 position0;
			vec3 direction0;
			if (light0posn.w != 0) {
				position0 = light0posn.xyz / light0posn.w ; 
				direction0 = normalize (position0 - mypos) ; // no attenuation 
			} else {
				position0 = light0posn.xyz;
				direction0 = normalize(position0);
			}
        vec3 half0 = normalize (direction0 + eyedirn) ;  
        col0 = ComputeLight(direction0, light0color, normal, half0, diffuse, specular, shininess) ;
		}

        // Light 1
		vec4 col1 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is1 != 0) {
			vec3 position1;
			vec3 direction1;
			if (light1posn.w != 0) {
				position1 = light1posn.xyz / light1posn.w ; 
				direction1 = normalize (position1 - mypos) ; // no attenuation 
			} else {
				position1 = light1posn.xyz;
				direction1 = normalize(position1);
			}
        vec3 half1 = normalize (direction1 + eyedirn) ;  
        col1 = ComputeLight(direction1, light1color, normal, half1, diffuse, specular, shininess) ;
		}
		
        // Light 2
		vec4 col2 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is2 != 0) {
			vec3 position2;
			vec3 direction2;
			if (light2posn.w != 0) {
				position2 = light2posn.xyz / light2posn.w ; 
				direction2 = normalize (position2 - mypos) ; // no attenuation 
			} else {
				position2 = light2posn.xyz;
				direction2 = normalize(position2);
			}
        vec3 half2 = normalize (direction2 + eyedirn) ;  
        col2 = ComputeLight(direction2, light2color, normal, half2, diffuse, specular, shininess) ;
		}

        // Light 3
		vec4 col3 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is3 != 0) {
			vec3 position3;
			vec3 direction3;
			if (light3posn.w != 0) {
				position3 = light3posn.xyz / light3posn.w ; 
				direction3 = normalize (position3 - mypos) ; // no attenuation 
			} else {
				position3 = light3posn.xyz;
				direction3 = normalize(position3);
			}
        vec3 half3 = normalize (direction3 + eyedirn) ;  
        col3 = ComputeLight(direction3, light3color, normal, half3, diffuse, specular, shininess) ;
		}

        // Light 4
		vec4 col4 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is4 != 0) {
			vec3 position4;
			vec3 direction4;
			if (light4posn.w != 0) {
				position4 = light4posn.xyz / light4posn.w ; 
				direction4 = normalize (position4 - mypos) ; // no attenuation 
			} else {
				position4 = light4posn.xyz;
				direction4 = normalize(position4);
			}
        vec3 half4 = normalize (direction4 + eyedirn) ;  
        col4 = ComputeLight(direction4, light4color, normal, half4, diffuse, specular, shininess) ;
		}

        // Light 5
		vec4 col5 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is5 != 0) {
			vec3 position5;
			vec3 direction5;
			if (light5posn.w != 0) {
				position5 = light5posn.xyz / light5posn.w ; 
				direction5 = normalize (position5 - mypos) ; // no attenuation 
			} else {
				position5 = light5posn.xyz;
				direction5 = normalize(position5);
			}
        vec3 half5 = normalize (direction5 + eyedirn) ;  
        col5 = ComputeLight(direction5, light5color, normal, half5, diffuse, specular, shininess) ;
		}

        // Light 6
		vec4 col6 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is6 != 0) {
			vec3 position6;
			vec3 direction6;
			if (light6posn.w != 0) {
				position6 = light6posn.xyz / light6posn.w ; 
				direction6 = normalize (position6 - mypos) ; // no attenuation 
			} else {
				position6 = light6posn.xyz;
				direction6 = normalize(position6);
			}
        vec3 half6 = normalize (direction6 + eyedirn) ;  
        col6 = ComputeLight(direction6, light6color, normal, half6, diffuse, specular, shininess) ;
		}

        // Light 7
		vec4 col7 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is7 != 0) {
			vec3 position7;
			vec3 direction7;
			if (light7posn.w != 0) {
				position7 = light7posn.xyz / light7posn.w ; 
				direction7 = normalize (position7 - mypos) ; // no attenuation 
			} else {
				position7 = light7posn.xyz;
				direction7 = normalize(position7);
			}
        vec3 half7 = normalize (direction7 + eyedirn) ;  
        col7 = ComputeLight(direction7, light7color, normal, half7, diffuse, specular, shininess) ;
		}

        // Light 8
		vec4 col8 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is8 != 0) {
			vec3 position8;
			vec3 direction8;
			if (light8posn.w != 0) {
				position8 = light8posn.xyz / light8posn.w ; 
				direction8 = normalize (position8 - mypos) ; // no attenuation 
			} else {
				position8 = light8posn.xyz;
				direction8 = normalize(position8);
			}
        vec3 half8 = normalize (direction8 + eyedirn) ;  
        col8 = ComputeLight(direction8, light8color, normal, half8, diffuse, specular, shininess) ;
		}

        // Light 9
		vec4 col9 = vec4(0.0, 0.0, 0.0, 0.0);
		if (is9 != 0) {
			vec3 position9;
			vec3 direction9;
			if (light9posn.w != 0) {
				position9 = light9posn.xyz / light9posn.w ; 
				direction9 = normalize (position9 - mypos) ; // no attenuation 
			} else {
				position9 = light9posn.xyz;
				direction9 = normalize(position9);
			}
        vec3 half9 = normalize (direction9 + eyedirn) ;  
        col9 = ComputeLight(direction9, light9color, normal, half9, diffuse, specular, shininess) ;
		}
        
        gl_FragColor = ambient + emission + col0 + col1 + col2 + col3 + col4 + col5 + col6 + col7 + col8 + col9 ; 
        }
}
