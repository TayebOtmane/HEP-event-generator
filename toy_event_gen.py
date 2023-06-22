import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings



class Particle(np.ndarray):
  '''defines the incoming particles (befor collision) along the z axis'''
  
    def __new__(cls, E, dir=1):
        obj = np.array([E, 0, 0, dir * E]).view(cls)
        obj.energy = E
        obj.dir = dir
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mass = getattr(obj, 'mass', None)
        self.energy = getattr(obj, 'energy', None)
        self.dir = getattr(obj, 'dir', None)



    def update(self):
        self[0] = self.energy
        self[3] = self.dir * self.energy
        return self

    def dot(self, fv2):
        dotp = self[1] * fv2[1] + self[2] * fv2[2] + self[3] * fv2[3] - self[0] * fv2[0]
        return dotp

    def lorentz_transform(self, beta, axis=3):
        gamma = 1 / np.sqrt(1 - beta**2)
        p_axis = gamma * (self[axis] - beta * self[0])
        p_0 = gamma * (self[0] - beta * self[axis])
        # self[axis] = p_axis
        self.energy = p_0
        self.update()
        return self


    def __repr__(self):
        return str(self)



###################################################################################################################

class Outgoing(np.ndarray):
  '''defines the outgoing particles (after collision)'''
    def __new__(cls, E, pvec):
        obj = np.array([E, pvec[0], pvec[1], pvec[2]]).view(cls)
        obj.energy = E
        obj.pvector = np.array(pvec)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.energy = getattr(obj, 'energy', None)
        self.pvector = getattr(obj, 'pvec', None)



    def dot(self, fv2):
        dotp = self[1] * fv2[1] + self[2] * fv2[2] + self[3] * fv2[3] - self[0] * fv2[0]
        return dotp

    def lorentz_transform(self, beta, axis=3):
        gamma = 1 / np.sqrt(1 - beta**2)
        p_axis = gamma * (self[axis] - beta * self[0])
        p_0 = gamma * (self[0] - beta * self[axis])
        self[axis] = p_axis
        self.energy = p_0
        return self


    def __repr__(self):
        return str(self)


###################################################################################################################

def collision(particle1, particle2, min_E=10, min_th=0.1):

    '''Generate random two four-vectors after colliding two particles along the z direction'''

    p_sum = particle1 + particle2
    if p_sum[3] != 0:
        beta = p_sum[3] / p_sum[0]
        print('This is beta:', beta)
        particle1 = particle1.lorentz_transform(beta)        
        particle2 = particle2.lorentz_transform(beta)
        print("This is the sum of p's boosted: ", particle1 + particle2)
        E_cm = particle1[0] + particle2[0]
    else:
        E_cm = p_sum[0]
    
    # The physics and dynamics are all in this little section (not yet at least) -----------
    E_lep = np.random.uniform(min_E, E_cm - min_E)
    E_jet = E_cm - E_lep

    phi_lep = np.random.uniform(0, 2 * np.pi)
    phi_jet = np.pi + phi_lep

    theta_lep = np.random.uniform(min_th, np.pi - min_th)
    theta_jet = np.arccos(-E_lep / E_jet * np.cos(theta_lep))

    # --------------------------------------------------------------------------------------
    xl = E_lep * np.sin(theta_lep) * np.cos(phi_lep)
    yl = E_lep * np.sin(theta_lep) * np.sin(phi_lep)
    zl = E_lep * np.cos(theta_lep)

    xj = E_jet * np.sin(theta_jet) * np.cos(phi_jet)
    yj = E_jet * np.sin(theta_jet) * np.sin(phi_jet)
    zj = E_jet * np.cos(theta_jet)
    
    vecl = np.array([xl, yl, zl])
    vecj = np.array([xj, yj, zj])

    out_lep = Outgoing(E_lep, vecl)
    out_jet = Outgoing(E_jet, vecj)
    
    return out_jet, out_lep


###################################################################################################################
# visualization of the three-momentum vectors

# Incoming particles
p1 = Particle(820.0, dir=-1)
p2 = Particle(27.439)

# Define vectors
vectors = [(p1[1:]), (p2[1:])]
counter = 0

# start collisions
num_of_events = 6
for i in range(num_of_events):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # Treat warnings as exceptions
        try:
            p3, p4 = collision(p1, p2)
            vectors.append(p3[1:])
            vectors.append(p4[1:])
        except RuntimeWarning:
            counter += 1

print("error counter is up to: ", counter)



# Create a new figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = ['red', 'green', 'black', 'yellow', 'pink', 'orange']

# Plot vectors
for vector in vectors[:2]:
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='blue', pivot='tip', normalize=False, arrow_length_ratio=0.1)
i = 0
j = 0
for vector in vectors[2:]:
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color[j], normalize=False, arrow_length_ratio=0.1)
    i += 1
    j += int(not(i%2))
    j = j%6


# Set limits and labels
scale = 200
ax.set_xlim([-scale, scale])
ax.set_ylim([-scale, scale])
ax.set_zlim([-scale, scale])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display the plot
plt.show()




