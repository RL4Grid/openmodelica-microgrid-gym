from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
#Initialize Model
m = GEKKO()

#define parameter

nomFreq = 50  # grid frequency / Hz
nomVolt = value=230
omega = 2*np.pi*nomFreq

J = 2
J_Q = 2000

R_lv_line_10km = 0
L_lv_line_10km = 0.00000083*10
B_L_lv_line_10km = -(omega * L_lv_line_10km)/(R_lv_line_10km**2 + (omega*L_lv_line_10km)**2)

R_load = 1
L_load = 1
G_RL_load = R_load/(R_load**2 + (omega*L_load)**2)
B_RL_load = -(omega * L_load)/(R_load**2 + (omega * L_load)**2)


B = np.array([[2*B_L_lv_line_10km, -B_L_lv_line_10km, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, 2*B_L_lv_line_10km+0, -B_L_lv_line_10km],
              [-B_L_lv_line_10km, -B_L_lv_line_10km, 2*B_L_lv_line_10km+B_RL_load]])

G = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, G_RL_load]])

#constants
w1 = 50
w2 = 50
w3 = 50
u1 = m.Var(value=230)
u2 = m.Var(value=230)
u3 = m.Var(value=230)
#p_offset = [0, 0, 0]

q_offset = [0, 0, 0]
#u1 = 230
#u2 = 230
#u3 = 230
#P1 = 1000
#P2 = 1000
#P3 = 2000

#P1 = m.Var(value=0)
#P2 = m.Var(value=0)
#P3 = m.Var(value=0)
Q1 = m.Var(value=0)
Q2 = m.Var(value=0)
Q3 = m.Var(value=0)
#w1 = m.Var(value=50)
#w2 = m.Var(value=50)
#w3 = m.Var(value=50)
theta1, theta2, theta3 = [m.Var() for i in range(3)]
#initialize variables

droop_linear=[-10000,-10000,0]
q_droop_linear=[-100000,-100000,0]
#initial values

theta1.value = 0
theta2.value = 0
theta3.value = 0
#w1.value = 50
#w2.value = 50
#w3.value = 50

    #Equations

#constraints

#m.Equation(u1 * u1 * (G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * (G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * (G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3)) == P1)
#m.Equation(u2 * u1 * (G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * (G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * (G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3)) == P2)
#m.Equation(u3 * u1 * (G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * (G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * (G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3)) == P3)

m.Equation(u1 * u1 * (G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) + \
           u1 * u2 * (G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) + \
           u1 * u3 * (G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3)) == Q1)
m.Equation(u2 * u1 * (G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) + \
           u2 * u2 * (G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) + \
           u2 * u3 * (G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3)) == Q2)
m.Equation(u3 * u1 * (G[2][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) + \
           u3 * u2 * (G[2][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) + \
           u3 * u3 * (G[2][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3)) == Q3)

# Equations

# define omega
m.Equation(theta1.dt()==w1)
m.Equation(theta2.dt()==w2)
m.Equation(theta3.dt()==w3)

#Power ODE

#m.Equation(J*w1*w1.dt()==(u1 * u1 * -(G[0][0] * m.cos(theta1 - theta1) + B[0][0] * m.sin(theta1 - theta1)) + \
#           u1 * u2 * -(G[0][1] * m.cos(theta1 - theta2) + B[0][1] * m.sin(theta1 - theta2)) + \
#           u1 * u3 * -(G[0][2] * m.cos(theta1 - theta3) + B[0][2] * m.sin(theta1 - theta3))))
#m.Equation(J*w2*w3.dt()==(u2 * u1 * -(G[1][0] * m.cos(theta2 - theta1) + B[1][0] * m.sin(theta2 - theta1)) + \
#           u2 * u2 * -(G[1][1] * m.cos(theta2 - theta2) + B[1][1] * m.sin(theta2 - theta2)) + \
#           u2 * u3 * -(G[1][2] * m.cos(theta2 - theta3) + B[1][2] * m.sin(theta2 - theta3))))
#m.Equation(J*w3*w3.dt()==(u3 * u1 * -(G[2][0] * m.cos(theta3 - theta1) + B[2][0] * m.sin(theta3 - theta1)) + \
#           u3 * u2 * -(G[2][1] * m.cos(theta3 - theta2) + B[2][1] * m.sin(theta3 - theta2)) + \
#           u3 * u3 * -(G[2][2] * m.cos(theta3 - theta3) + B[2][2] * m.sin(theta3 - theta3))))

#m.Equation(J*w1*w1.dt()==((-P1+p_offset[0])+(droop_linear[0]*(w1-nomFreq)))/(J*w1))
#m.Equation(J*w2*w3.dt()==((-P2+p_offset[1])+(droop_linear[1]*(w2-nomFreq)))/(J*w2))
#m.Equation(J*w3*w3.dt()==((-P3+p_offset[2])+(droop_linear[2]*(w3-nomFreq)))/(J*w3))

#Q_ODE


#m.Equation(J_Q*w1*w1.dt()==(u1 * u1 * -(G[0][0] * m.sin(theta1 - theta1) + B[0][0] * m.cos(theta1 - theta1)) + \
#           u1 * u2 * -(G[0][1] * m.sin(theta1 - theta2) + B[0][1] * m.cos(theta1 - theta2)) + \
#           u1 * u3 * -(G[0][2] * m.sin(theta1 - theta3) + B[0][2] * m.cos(theta1 - theta3))))
#m.Equation(J_Q*w2*w3.dt()==(u2 * u1 * -(G[1][0] * m.sin(theta2 - theta1) + B[1][0] * m.cos(theta2 - theta1)) + \
#           u2 * u2 * -(G[1][1] * m.sin(theta2 - theta2) + B[1][1] * m.cos(theta2 - theta2)) + \
#           u2 * u3 * -(G[1][2] * m.sin(theta2 - theta3) + B[1][2] * m.cos(theta2 - theta3))))
#m.Equation(J_Q*w3*w3.dt()==(u3 * u1 * -(G[2][0] * m.sin(theta3 - theta1) + B[2][0] * m.cos(theta3 - theta1)) + \
#           u3 * u2 * -(G[2][1] * m.sin(theta3 - theta2) + B[2][1] * m.cos(theta3 - theta2)) + \
#           u3 * u3 * -(G[2][2] * m.sin(theta3 - theta3) + B[2][2] * m.cos(theta3 - theta3))))



m.Equation(u1.dt()==((Q1+q_offset[0])+(q_droop_linear[0]*(u1-nomVolt)))/(J_Q*u1))
m.Equation(u2.dt()==((Q2+q_offset[1])+(q_droop_linear[1]*(u2-nomVolt)))/(J_Q*u2))
m.Equation(u3.dt()==((Q3+q_offset[2])+(q_droop_linear[2]*(u3-nomVolt)))/(J_Q*u3))

#m.Equation(J_Q*u1*u1.dt()==(-Q1))
#m.Equation(J_Q*u2*u2.dt()==(-Q2))
#m.Equation(J_Q*u3*u3.dt()==(-Q3))


#Set global options
m.options.IMODE = 7 #steady state optimization

m.time = np.linspace(0,100,500) # time points



#Solve simulation
m.solve()

#Results

#plt.plot(m.time,w1)
#plt.xlabel('time')
#plt.ylabel('w1(t)')


#plt.plot(m.time,w2)
#plt.xlabel('time')
#plt.ylabel('w2(t)')


#plt.plot(m.time,w3,'--')
#plt.xlabel('time')
#plt.ylabel('w3(t)')
#plt.ylim(48, 52)
#plt.show()



plt.plot(m.time,u1, 'b')
plt.xlabel('time')
plt.ylabel('u1(t)')

plt.plot(m.time,u2, '--r')
plt.xlabel('time')
plt.ylabel('u2(t)')

plt.plot(m.time,u3,'--g')
plt.xlabel('time')
plt.ylabel('u3(t)')
#plt.ylim(48, 52)
plt.show()



#plt.plot(m.time,P1)
#plt.plot(m.time,P2)
#plt.plot(m.time,P3)
#plt.xlabel('time')
#plt.ylabel('P(t)')
#plt.show()


#plt.plot(m.time,Q1)
#plt.plot(m.time,Q2)
#plt.plot(m.time,Q3)
#plt.xlabel('time')
#plt.ylabel('Q(t)')
#plt.show()


plt.plot(m.time,(np.array(theta1.value)-np.array(theta2.value)))
plt.plot(m.time,(np.array(theta1.value)-np.array(theta3.value)))
plt.plot(m.time,(np.array(theta2.value)-np.array(theta3.value)))
#plt.legend()
plt.xlabel('time')
plt.ylabel('diff_theta(t)')
plt.show()