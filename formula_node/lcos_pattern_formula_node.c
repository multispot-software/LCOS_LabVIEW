int32 image_width, image_height;
float64 pixel_pitch;
float64 pattern_pitch;
float64 focal_distance;
float64 focal_distance_fresn;
float64 focal_distance_min, focal_distance_difflim;
float64 cos_theta;
float64 pat_cent_x, pat_cent_y;
float64 x_dist, y_dist, dist_sqr;
float64 patnumx, patnumy;
float64 patnumsqr, patnumscale, patnumratio;
float64 rotx, roty;
float64 rotm00, rotm01, rotm10, rotm11;
float64 val;
float64 pattern_spotsize;
float64 patnumx_frac, patnumy_frac;
float64 pattern_spotborder, pattern_spotbordertop, pattern_spotborderbot;
float64 quadratic_adj;
float64 wrap_dist, focal_shift, focal_perspot;
float64 pat_xcent, pat_ycent;
float64 x0_folcalshift, y0_folcalshift;
int32 corrheight, corrwidth;
int32 first_xspot, first_yspot, last_xspot, last_yspot;
int32 x, y, d;
int32 LVbug_cx, LVbug_cy;
int32 patintx, patinty;

// USER CONSTANTS
image_width = 800;
image_height = 600;
pixel_pitch = 20e-6;

// Pattern size/geometry
pattern_pitch = pattern_spacing_in_pixels * pixel_pitch;
pattern_spotsize = pattern_spotsize_in_pixels*pixel_pitch;

// Coordinates of LCOS center in pixel units
pat_cent_x = image_width / 2;
pat_cent_y = image_height / 2;

// XY numeration of spots. Ex. num_spot=5: first = -2; last = 2
//							   num_spot=4: first = -2; last = 2
// num_xspot, num_yspot: user provided (int32)
// NOTE: LV truncate to int towards 0
first_xspot = -num_xspots/2; 
last_xspot = first_xspot + num_xspots;
first_yspot = -num_yspots/2; 
last_yspot = first_yspot + num_yspots;
LVbug_cx = 0;
LVbug_cy = 0;

// - - - SPOT BORDER - - - (useful when spot_size , spot_pitch)
// Spot border width in percentage of pattern_pitch (or pattern_pitch units)
pattern_spotborder = 0.5*(1 - pattern_spotsize/pattern_pitch);
// Border start coordinate on top/bottom in pattern_pitch units
pattern_spotbordertop = 0.5 - pattern_spotborder;
pattern_spotborderbot = -0.5 + pattern_spotborder;

// Rescale pattern translation/rotation in standard units
pattern_offset_x *= pixel_pitch;                   // mod XM
pattern_offset_y *= pixel_pitch;                  // mod XM
rot_angle *= (pi/180);                                 // mod XM

// Use user-provided focal distance
focal_distance = focal_set * 1e-3;

// Recenter odd numbered spots.
if (num_xspots & 1) {
	pattern_offset_x -= 0.5*pattern_pitch * (1 - (num_xspots & 2));
	LVbug_cx = -(num_xspots & 2) / 2;
}
if (num_yspots & 1) {
	pattern_offset_y -= 0.5*pattern_pitch * (1 - (num_yspots & 2));
	LVbug_cy = -(num_yspots & 2) / 2;
}

// Prepare rotation matrix.
rotm00 = cos(rot_angle);
rotm01 = sin(rot_angle);
rotm11 = rotm00;
rotm10 = -rotm01;

// Calculate pattern
for (y=0; y<image_height; y++) {
	for (x=0; x<image_width; x++) {
		// Calculate pattern in rotated space.
		// rotx, roty: rotated&centered coordinates in pixel units
		rotx = rotm00 * (x-pat_cent_x) + rotm01 * (y-pat_cent_y);
		roty = rotm10 * (x-pat_cent_x) + rotm11 * (y-pat_cent_y);

		// Square distance from pattern center in pattern_pitch (spot number) units
		patnumsqr = (pixel_pitch**2)*(rotx**2 + roty**2)/(pattern_pitch**2);

		// Rotated&centered coordinates in spot-number units
		patnumx = (rotx*pixel_pitch-pattern_offset_x)/pattern_pitch;
		patnumy = (roty*pixel_pitch-pattern_offset_y)/pattern_pitch;

		// Center of current spot in meters
		pat_xcent = pattern_pitch*(floor(patnumx) + 0.5);
		pat_ycent = pattern_pitch*(floor(patnumy) + 0.5);
		
		// Coordinates relative to the spot center in spot-number units
		patnumx_frac = patnumx - (floor(patnumx) + 0.5);
		patnumy_frac = patnumy - (floor(patnumy) + 0.5);

		// If this spot number is in the range.
		if ((patnumx >= first_xspot) && (patnumx < last_xspot) &&
			(patnumy >= first_yspot) && (patnumy < last_yspot) &&
			(patnumx_frac >= pattern_spotborderbot) &&
			(patnumx_frac <= pattern_spotbordertop) &&
			(patnumy_frac >= pattern_spotborderbot) &&
			(patnumy_frac <= pattern_spotbordertop) ) {
			
			x0_folcalshift = pattern_pitch*(floor(patnumx)-xo_focalshift);
			y0_folcalshift = pattern_pitch*(floor(patnumy)+yo_focalshift);
			focal_shift = focal_shiftparam*(x0_folcalshift**2 + y0_folcalshift**2);
			focal_perspot = focal_distance + focal_shift;
			
			if (darken_center_spot &&
				patnumx < 1+LVbug_cx && patnumx > LVbug_cx &&
				patnumy < 1+LVbug_cy && patnumy > LVbug_cy) {
				lcospat[y][x] = 0;
				continue;
			}
			
			// Coordinates relative to the spot center in meters
			x_dist = pattern_pitch * patnumx_frac;
			y_dist = pattern_pitch * patnumy_frac;

			dist_sqr = x_dist*x_dist + y_dist*y_dist;

			if (exact_formula) {
				cos_theta = cos(atan(sqrt(dist_sqr)/focal_perspot));
				lcospat[y][x] = const_phase - 1e-6 - (2/wavelength) * (1-cos_theta) * sqrt(dist_sqr + focal_perspot**2); 
			} 
			else
				lcospat[y][x] = const_phase - 1e-6 - dist_sqr / (wavelength*focal_distance);
		}
		else {
		// Make unused area plane wave, to be filtered out by pindot.
		lcospat[y][x] = 0;
		}
	}
}

// Steering specular reflection out of the field of view
for (y=0; y<image_height; y++) {
	for (x=0; x<image_width; x++) {
		d = hv_steering ? y: x;
		steer_pattern[y][x] = (d - lw*floor(d/lw))/(lw-1)*vmax_steering;
	}
}

// Map to greyscale values
for (y=0; y<image_height; y++) {
	for (x=0; x<image_width; x++) {
		val = lcospat[y][x];
		
		// Set the outside of the pattern for beam steering
		if ((val == 0) && (lw > 1))
			val = steer_pattern[y][x]/phase_factor;

		// IF phase doesn't wrap
		// set pixels in between pattern to 0 for smoothness
		if ((!wrap_phase) && (val < 0))
			val = 0;
		
		// ELSE remove extra 2PI
		if (wrap_phase) {
			if ((val < 0) || (!phase_overshoot && (val >= 2)))
				val -= floor(val/2)*2;
		}

		// Map PI to correct greyscale value.
		val = val * phase_factor;
		
		lcospat[y][x] = val;
	}
}

