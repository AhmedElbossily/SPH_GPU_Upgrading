// Copyright ETH Zurich, IWF

// This file is part of iwf_mfree_gpu_3d.

// iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "solid_state_processes.h"

particle_gpu *setup_RFSSW(int nbox, grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	joco_constants joco = make_joco_constants();

	float_t vel_cylinders = 280 * 1000; // mm/s
	float_t ri = 35;
	float_t ro = 40;
	float_t height = (ro - ri);
	float_t spacing = ro + 1;

	float_t dx = 2 * ro / (nbox - 1);
	float_t hdx = 1.7;

	phys.E = 71.7e9;
	phys.nu = 0.33;
	phys.rho0 = 2830.0 * 1.0e-6;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dx * dx * dx * phys.rho0;

	// Johnson Cook Constants substrate
	joco.A = 450.821e6; //	450.821 MPa
	joco.B = 0.;		//	108.537 MPA
	joco.C = 0.027;		//	0.027
	joco.m = 0.981;		//	0.981
	joco.n = 0.;		//	0.045
	joco.Tref = 20.;	//	323
	joco.Tmelt = 630.;	//	488
	joco.eps_dot_ref = 1;
	joco.clamp_temp = 1.;

	trml_wp.cp = 860. * 1.0e6;							  // Heat Capacity
	trml_wp.tq = 0.9;									  // Taylor-Quinney Coefficient
	trml_wp.k = 153. * 1.0e6;							  // Thermal Conduction
	trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // Thermal diffusivity
	trml_wp.eta = 0.9;

	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.5;

	corr.stresseps = 0.3;
	{
		float_t h1 = 1. / (hdx * dx);
		float_t q = dx * h1;
		float_t fac = (M_1_PI)*h1 * h1 * h1;
		;
		corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));
	}

	int nheight = height / dx;

	std::vector<float4_t> pos;
	for (int i = 0; i < nbox; i++)
	{
		for (int j = 0; j < nbox; j++)
		{
			for (int k = 0; k < nheight; k++)
			{
				float_t px = -ro + i * dx;
				float_t py = -ro + j * dx;
				float_t pz = k * dx;
				float_t dist = sqrt(px * px + py * py);
				if (dist < ro && dist >= ri)
				{
					float4_t posl, posr;

					posl.x = px - spacing;
					posl.y = py;
					posl.z = pz;

					posr.x = px + spacing;
					posr.y = py;
					posr.z = pz;

					pos.push_back(posl);
					pos.push_back(posr);
				}
			}
		}
	}

	int n = pos.size();

	global_time_dt = 1e-7 * 0.3;
	global_time_final = ro / vel_cylinders * 3;

	*grid = new grid_gpu_green(n, make_float3_t(-200, -60, -60), make_float3_t(+200, +60, +60), hdx * dx);

	printf("calculating with %d\n", n);

	float4_t *vel = new float4_t[n];
	float_t *h = new float_t[n];
	float_t *rho = new float_t[n];
	float_t *T = new float_t[n];
	float_t *tool_p = new float_t[n];
	float_t *fixed = new float_t[n];

	for (int i = 0; i < n; i++)
	{
		rho[i] = phys.rho0;
		h[i] = hdx * dx;
		vel[i].x = (pos[i].x < 0) ? vel_cylinders : -vel_cylinders;
		vel[i].y = 0.;
		vel[i].z = 0.;
		T[i] = joco.Tref;
		tool_p[i] = 0.0;
		fixed[i] = 0.0;
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_johnson_cook_constants(joco);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_thermal_constants_workpiece(trml_wp);

	float4_t *pos_f = new float4_t[n];
	for (int i = 0; i < n; i++)
	{
		pos_f[i] = pos[i];
	}
	particle_gpu *particles = new particle_gpu(pos_f, vel, rho, T, h, fixed, tool_p, n);

	assert(check_cuda_error());
	return particles;
}
