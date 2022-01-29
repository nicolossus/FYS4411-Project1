import numpy as np
import random

class Particle:
    def __init__(self, position):
        self.position = position



class System:
    def __init__(self, seed, omega_HO=1.0):
        self.seed = seed
        self.omega_HO = omega_HO
        self.a = 0.0


    def set_number_particles(self, num_particles):
        self.numberOfParticles = num_particles
        self.particles = []
        for i in range(self.numberOfParticles):
            particle = Particle(np.zeros(self.numberOfDimensions))
            self.particles.append(particle)


    def set_number_dimensions(self, num_dimensions):
        self.numberOfDimensions = num_dimensions

    def initialize_system(self):
        random.seed(self.seed)
        particles = []
        for i in range(self.numberOfParticles):
            position = np.random.uniform(0, 1, size=(self.numberOfDimensions))
            particle = Particle(position)
            particles.append(particle)
        self.particles = particles

    def calculate_energy(self, alpha, beta):
        kinetic_E, potential_E = self.HO_Hamiltionian(alpha, beta)
        local_energy = kinetic_E + potential_E
        #print("Potential energy: ", potential_E)
        #print("Kinetic energy: ", kinetic_E)
        #print("Local energy: ", local_energy)
        return local_energy

    def Gaussian(self, alpha, beta):
        num_particles = self.numberOfParticles
        num_dimensions = self.numberOfDimensions
        particles = self.particles
        gauss_val = 1
        for i in range(num_particles):
            r = particles[i].position
            for j in range(num_dimensions):
                if j==2:
                    gauss_val *= np.exp(-alpha*beta*r[j]*r[j])
                else:
                    gauss_val *= np.exp(-alpha*r[j]*r[j])
            for k in range(i):
                    rik = r-particles[k].position
                    norm_rik = np.linalg.norm(rik)
                    gauss_val *= np.exp(self.u(norm_rik))

        return gauss_val

    def u(self, norm_r):
        a = self.a
        if norm_r <= a:
            return -1.0e10
        else:
            return 1-a/norm_r

    def u_der(self, norm_r):
        a = self.a
        if norm_r <= a:
            return 0.0
        else:
            return a/(norm_r*norm_r-norm_r*a)

    def u_der2(self, norm_r):
        a = self.a
        if norm_r <= a:
            return 0.0
        else:
            return (a*a-2*norm_r*a)/((norm_r*norm_r - norm_r*a)**2)


    def HO_Hamiltionian(self, alpha, beta):
        kinetic_energy = 0
        potential_energy = 0
        particles = self.particles
        num_particles = self.numberOfParticles
        num_dimensions = self.numberOfDimensions
        for i in range(num_particles):
            particle = particles[i]
            r = particle.position
            # laplacian phi(r_i) / phi(r_i)
            kinetic_energy += -2*alpha*alpha*np.dot(r,r) + alpha
            # gradient phi(r_i)/phi(r_i)sum_j/=i[rij/norm(rij)u'(rij)]
            for j in range(num_particles):
                if j==i:
                    kinetic_energy += 0.0
                else:
                    rij = r-particles[j].position
                    norm_rij = np.linalg.norm(rij)
                    kinetic_energy += alpha*r*rij/norm_rij*self.u_der(norm_rij)
            # -0.5sum_j/=i, sum_k/=i [rij*rik/(norm(rij)*norm(rik))*u'(rij)*u'(rik)]
            for j in range(num_particles):

                if j==i:
                    kinetic_energy += 0.0
                else:
                    rij = r-particles[j].position
                    norm_rij = np.linalg.norm(rij)
                    for k in range(num_particles):
                        rik = r-particles[k].position
                        norm_rik = np.linalg.norm(rik)
                        if k==i:
                            kinetic_energy += 0.0
                        else:
                            kinetic_energy -= 0.5*np.dot(rij, rik)/(norm_rij*norm_rik)*self.u_der(norm_rij)*self.u_der(norm_rik)
            # -0.5*sum_j/=i [u''(rij)+2u'(rij)/norm_rij]
            for j in range(num_particles):
                if j==i:
                    kinetic_energy += 0.0
                else:
                    rij = r-particles[j].position
                    norm_rij = np.linalg.norm(rij)
                    kinetic_energy -= 0.5*(self.u_der2(norm_rij)+2.0*self.u_der(norm_rij)/norm_rij)

            potential_energy += 0.5*self.omega_HO*self.omega_HO*np.dot(r,r)

        kin_E = np.sum(kinetic_energy)
        pot_E = np.sum(potential_energy)
        return kin_E, pot_E


    def update_positions(self, positions):
        num_particles = self.numberOfParticles
        for i in range(num_particles):
            self.particles[i].position = positions[i]

    def Metropolis(self):
        file = open("Jørn/metropolis.dat", "w")
        cycles = 100
        alpha_start = 0.2
        positions_old = np.zeros((self.numberOfParticles, self.numberOfDimensions), np.double)
        positions_new = np.zeros((self.numberOfParticles, self.numberOfDimensions), np.double)
        alpha_step_size = 0.1
        beta_step_size = 0.1
        step_size = 1.0
        np.random.seed(self.seed)
        max_variations = 7
        alpha_values = np.zeros(max_variations)
        beta_values = np.zeros(max_variations)
        energies = np.zeros((max_variations, max_variations), np.double)
        variances = np.zeros((max_variations, max_variations), np.double)
        errors = np.zeros((max_variations, max_variations), np.double)
        for ia in range(max_variations):
            alpha = alpha_start + ia*alpha_step_size
            alpha_values[ia] = alpha
            beta_start = 0.7
            for ib in range(max_variations):
                beta = beta_start + ib*beta_step_size
                beta_values[ib] = beta
                energy = energy2 = 0.0
                for i in range(self.numberOfParticles):
                    for j in range(self.numberOfDimensions):
                        positions_old[i,j] = step_size*(random.random()-0.5)
                        self.update_positions(positions_old)
                wfold = self.Gaussian(alpha,beta)
                for cycle in range(cycles):
                    for i in range(self.numberOfParticles):
                        for j in range(self.numberOfDimensions):
                            positions_new[i,j] = step_size*(random.random()-0.5)
                            self.update_positions(positions_new)
                        wfnew = self.Gaussian(alpha, beta)

                        if random.random() < wfnew**2/wfold**2:
                            for j in range(self.numberOfDimensions):
                                positions_old[i,j] = positions_new[i,j]
                            wfold = wfnew
                    energy += self.calculate_energy(alpha, beta)
                    energy2 += self.calculate_energy(alpha, beta)*self.calculate_energy(alpha, beta)
                energy /= cycles
                energy2 /= cycles
                variance = energy2-energy*energy
                error = np.sqrt(variance/cycles)
                energies[ia, ib] = energy
                variances[ia, ib] = variance
                errors[ia, ib] = error
                file.write('%f %f %f %f %f\n' %(alpha,beta,energy,variance,error))
        file.close()
        return energies, variances,alpha_values, beta_values







system = System(101, omega_HO=1.0)
system.set_number_dimensions(1)
system.set_number_particles(10)
#system.initialize_system()
#print(system.numberOfParticles)
#system.calculate_energy(0.5, 1.0)
Es, Vs, alphas, betas = system.Metropolis()
print(Es)
