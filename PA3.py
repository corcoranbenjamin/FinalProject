import numpy as np #numpy for all math 
import matplotlib.pyplot as plt #matplotlib is for plotting
from matplotlib.patches import Circle, Polygon #brings in circle and polygon for the plot
from scipy.integrate import odeint #for ODE simulation

#************************ VARIABLES ***************************
#link lengths and point masses
L1, L2 = 0.5, 0.3
m1, m2 = 1, 1 

#moment of inertia and center of mass
I1, I2 = 1/12, 1/12
a1, a2 = L1/2, L2/2 

#simulation parameters
gravity = 9.81 
tStep = 0.01

#initialize flag so animation only runs once
animationRunning = False

#pause time at waypoints and end of animation
pauseTime = 0.2       #brief pause at corners (square/triangle)
endPauseTime = 1.0    #longer pause at the end of animation  

#define workspace annulus
minWS = abs(L1 - L2)  
maxWS = L1 + L2     

#joint limits and maximum cartesian velocity
jointLimits = [np.deg2rad(-90), np.deg2rad(90)]
maxCartesianVelocity = 0.15  #meters per second

#notes:
#increased kp until there was no steady state error
#started with Kd at 10 for both joints
#increased kd of first joint to prevent overshoot
#best results: [15, 50], [50, 10]
#controller gains for joints 1 and 2
Kp = np.array([15, 50])  
Kd = np.array([100, 10])    

#shape center and size
shapeCenter = [0.5, 0.3]  #center of shape in workspace
shapeSize = 0.2  #size of shape (radius for circle, side length for square/triangle)


#************************ FUNCTIONS ***************************
#function returns theta 1 and 2 for a given x, y position
#input: x, y (meters)
#output: theta1, theta2 (radians)
def inverseKinematics(x, y): 
    
    #calculate theta 1, 2 using the inverse kinematics equations
    theta2 = np.acos((x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2))
    theta1 = np.arctan2(y, x) - np.arctan2(L2*np.sin(theta2), L1 + L2*np.cos(theta2))
    
    return theta1, theta2 


#function calulates the joint coordinates given theta1 and theta2
#input: theta1, theta2 (radians)
#output: x1, y1, x2, y2 (meters)
def forwardKinematics(theta1, theta2):
 
    #calculate joint positions for animation using forward kinematics
    x1 =  L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = L2 * np.cos(theta1 + theta2) + x1
    y2 = L2 * np.sin(theta1 + theta2) + y1

    return x1, y1, x2, y2


#generates waypoints for the selected shape
#input: shape (string: 'oval', 'square', 'triangle'), center [x, y], size (radius or side length)
#output: list of waypoints [[x1, y1], [x2, y2], ...], isContinuous (bool)
def generateShapeWaypoints(shape, center, size):
    
    waypoints = []
    cx, cy = center
    isContinuous = False  #flag for shapes that need smooth continuous motion
    
    if shape == 'oval':
        #oval is a continuous smooth shape - flag it for special trajectory handling
        isContinuous = True
        #oval dimensions (wider than tall)
        a = size      #semi-major axis (horizontal)
        b = size * 0.6  #semi-minor axis (vertical)
        numPoints = 60  #more points for smoother curve
        for i in range(numPoints + 1):
            angle = 2 * np.pi * i / numPoints
            x = cx + a * np.cos(angle)
            y = cy + b * np.sin(angle)
            waypoints.append([x, y])
            
    elif shape == 'square':
        #generate 4 corners of square, starting from top-right going clockwise
        halfSize = size / 2
        waypoints = [
            [cx + halfSize, cy + halfSize],  #top right
            [cx - halfSize, cy + halfSize],  #top left
            [cx - halfSize, cy - halfSize],  #bottom left
            [cx + halfSize, cy - halfSize],  #bottom right
            [cx + halfSize, cy + halfSize]   #back to start
        ]
        
    elif shape == 'triangle':
        #generate equilateral triangle vertices, pointing up
        height = size * np.sqrt(3) / 2
        waypoints = [
            [cx, cy + 2*height/3],              #top vertex
            [cx - size/2, cy - height/3],       #bottom left
            [cx + size/2, cy - height/3],       #bottom right
            [cx, cy + 2*height/3]               #back to start
        ]
    
    return waypoints, isContinuous


#generates smooth continuous trajectory for oval/circular shapes
#input: waypoints [[x1, y1], ...], totalTime (seconds)
#output: trajectory [(theta1, theta2, thetaDot1, thetaDot2, thetaDoubleDot1, thetaDoubleDot2)]
def generateContinuousTrajectory(waypoints, totalTime):
    
    trajectory = []
    numPoints = len(waypoints) - 1  #exclude duplicate end point
    
    t = 0
    while t < totalTime:
        #parameter s goes from 0 to numPoints as t goes from 0 to totalTime
        s = (t / totalTime) * numPoints
        
        #find which segment we're in
        idx = int(s) % numPoints
        localS = s - int(s)  #fractional part [0, 1)
        
        #get current and next waypoint
        p0 = np.array(waypoints[idx])
        p1 = np.array(waypoints[(idx + 1) % numPoints])
        
        #linear interpolation for position
        pos = p0 + localS * (p1 - p0)
        x, y = pos[0], pos[1]
        
        #velocity: derivative of position with respect to time
        #ds/dt = numPoints / totalTime
        dsdt = numPoints / totalTime
        dpos_ds = p1 - p0  #change in position per unit s
        vel = dpos_ds * dsdt
        xDot, yDot = vel[0], vel[1]
        
        #acceleration is zero for constant velocity along each segment
        xDoubleDot, yDoubleDot = 0.0, 0.0
        
        #convert to joint space using inverse kinematics
        theta1, theta2 = inverseKinematics(x, y)
        
        #compute Jacobian for velocity conversion
        J11 = -L1*np.sin(theta1) - L2*np.sin(theta1 + theta2)
        J12 = -L2*np.sin(theta1 + theta2)
        J21 = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
        J22 = L2*np.cos(theta1 + theta2)
        J = np.array([[J11, J12], [J21, J22]])
        
        #convert cartesian velocity to joint velocity
        cartVel = np.array([xDot, yDot])
        thetaDot = np.linalg.solve(J, cartVel)
        thetaDot1, thetaDot2 = thetaDot[0], thetaDot[1]
        
        #compute Jacobian derivative for acceleration
        Jdot11 = -L1*np.cos(theta1)*thetaDot1 - L2*np.cos(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot12 = -L2*np.cos(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot21 = -L1*np.sin(theta1)*thetaDot1 - L2*np.sin(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot22 = -L2*np.sin(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot = np.array([[Jdot11, Jdot12], [Jdot21, Jdot22]])
        
        #convert cartesian acceleration to joint acceleration
        cartAccel = np.array([xDoubleDot, yDoubleDot])
        thetaDoubleDot = np.linalg.solve(J, cartAccel - Jdot @ thetaDot)
        thetaDoubleDot1, thetaDoubleDot2 = thetaDoubleDot[0], thetaDoubleDot[1]
        
        trajectory.append((theta1, theta2, thetaDot1, thetaDot2, thetaDoubleDot1, thetaDoubleDot2))
        t += tStep
    
    #add longer pause at the end of the shape
    finalTheta1, finalTheta2 = inverseKinematics(waypoints[0][0], waypoints[0][1])
    t = 0
    while t < endPauseTime:
        trajectory.append((finalTheta1, finalTheta2, 0, 0, 0, 0))
        t += tStep
    
    return trajectory


#calculates the total duration of cartesian trajectory segment
#input: initial and final positions [x, y] (meters)
#output: total duration of trajectory (seconds)
def cartesianTrajectoryDuration(initialPos, finalPos):
    
    #calculate distance between initial and final positions
    distance = np.sqrt((finalPos[0] - initialPos[0])**2 + (finalPos[1] - initialPos[1])**2)
    
    #determine duration based on max cartesian velocity
    totalDuration = distance / maxCartesianVelocity
    
    #ensure minimum duration for very short segments
    if totalDuration < 0.1:
        totalDuration = 0.1
    
    return totalDuration


#takes time within trajectory segment and returns position, velocity, acceleration for one axis
#input: t (seconds), posInitial (meters), posFinal (meters), totalDuration (seconds)
#output: pos (meters), vel (m/s), accel (m/s**2)
def currentCartesianTrajectory(t, posInitial, posFinal, totalDuration):

    #calculate coefficients for trajectory cubic (same cubic polynomial for smooth motion)
    c0 = posInitial
    c1 = 0
    c2 = 3*(posFinal - posInitial)/(totalDuration**2)
    c3 = -2*(posFinal - posInitial)/(totalDuration**3)

    #generate cartesian states
    pos = c0 + c1*t + c2*t**2 + c3*t**3
    vel = c1 + 2*c2*t + 3*c3*t**2
    accel = 2*c2 + 6*c3*t

    return pos, vel, accel


#generate cartesian trajectory segment between two waypoints (straight line in cartesian space)
#input: initialPos [x, y] (meters), finalPos [x, y] (meters)
#output: trajectory [(theta1, theta2, thetaDot1, thetaDot2, thetaDoubleDot1, thetaDoubleDot2)]
def generateCartesianTrajectory(initialPos, finalPos):
    
    #create empty list to store trajectory
    trajectory = []
    totalDuration = cartesianTrajectoryDuration(initialPos, finalPos)

    t = 0

    #generate cartesian trajectory and convert to joint space via IK
    while t < totalDuration:
        #get cartesian position, velocity, acceleration at time t
        x, xDot, xDoubleDot = currentCartesianTrajectory(t, initialPos[0], finalPos[0], totalDuration)
        y, yDot, yDoubleDot = currentCartesianTrajectory(t, initialPos[1], finalPos[1], totalDuration)
        
        #convert cartesian position to joint angles using inverse kinematics
        theta1, theta2 = inverseKinematics(x, y)
        
        #compute Jacobian for velocity and acceleration conversion
        J11 = -L1*np.sin(theta1) - L2*np.sin(theta1 + theta2)
        J12 = -L2*np.sin(theta1 + theta2)
        J21 = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
        J22 = L2*np.cos(theta1 + theta2)
        J = np.array([[J11, J12], [J21, J22]])
        
        #convert cartesian velocity to joint velocity: thetaDot = J^-1 * xDot
        cartVel = np.array([xDot, yDot])
        thetaDot = np.linalg.solve(J, cartVel)
        thetaDot1, thetaDot2 = thetaDot[0], thetaDot[1]
        
        #compute Jacobian derivative for acceleration conversion
        Jdot11 = -L1*np.cos(theta1)*thetaDot1 - L2*np.cos(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot12 = -L2*np.cos(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot21 = -L1*np.sin(theta1)*thetaDot1 - L2*np.sin(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot22 = -L2*np.sin(theta1 + theta2)*(thetaDot1 + thetaDot2)
        Jdot = np.array([[Jdot11, Jdot12], [Jdot21, Jdot22]])
        
        #convert cartesian acceleration to joint acceleration: thetaDoubleDot = J^-1 * (xDoubleDot - Jdot * thetaDot)
        cartAccel = np.array([xDoubleDot, yDoubleDot])
        thetaDoubleDot = np.linalg.solve(J, cartAccel - Jdot @ thetaDot)
        thetaDoubleDot1, thetaDoubleDot2 = thetaDoubleDot[0], thetaDoubleDot[1]
        
        trajectory.append((theta1, theta2, thetaDot1, thetaDot2, thetaDoubleDot1, thetaDoubleDot2))
        t += tStep

    t = 0
    
    #add pause time at waypoint
    finalTheta1, finalTheta2 = inverseKinematics(finalPos[0], finalPos[1])
    while t < pauseTime:
        trajectory.append((finalTheta1, finalTheta2, 0, 0, 0, 0))
        t += tStep
    
    return trajectory


#generate full trajectory through all waypoints
#input: waypoints [[x1, y1], [x2, y2], ...]
#output: full trajectory as numpy array
def generateFullTrajectory(waypoints):
    
    fullTrajectory = []
    
    #generate trajectory between each consecutive pair of waypoints
    for i in range(len(waypoints) - 1):
        segment = generateCartesianTrajectory(waypoints[i], waypoints[i+1])
        fullTrajectory.extend(segment)
    
    #add longer pause at the very end of animation
    finalPos = waypoints[-1]
    finalTheta1, finalTheta2 = inverseKinematics(finalPos[0], finalPos[1])
    t = 0
    while t < endPauseTime:
        fullTrajectory.append((finalTheta1, finalTheta2, 0, 0, 0, 0))
        t += tStep
    
    return fullTrajectory

#defines the equations of motion and control law for the robot
#input: statevar (current state of the robot), t (current time), theta1Desired (desired theta1), theta2Desired (desired theta2), thetaDot1Desired (desired thetaDot1), thetaDot2Desired (desired thetaDot2), tSpan (time span), Kp (proportional gain), Kd (derivative gain)
#output: joint accelerations for both joints
def manipulatorEOM(statevar, t, theta1Desired, theta2Desired, thetaDot1Desired, thetaDot2Desired, tSpan, Kp, Kd):
    
    #get the current state 
    theta1 = statevar[0]
    theta2 = statevar[1]
    thetaDot1 = statevar[2]
    thetaDot2 = statevar[3]

    #subtract current time from each element in tSpan. Find the minimum and that is closest to the current time
    i = np.argmin(np.abs(tSpan - t)) 
   
    #Mass Matrix
    M11 = I1 + I2 + m1*a1**2 + m2*(L1**2 + a2**2 + 2*L1*a2*np.cos(theta2))
    M12 = I2 + m2*(a2**2 + L1*a2*np.cos(theta2))
    M21 = M12
    M22 = I2 + m2*a2**2
    M = np.array([[M11, M12], [M21, M22]])
    
    #Coriolis/Centrifugal forces
    C1 = -m2*L1*a2*np.sin(theta2)*(2*thetaDot1*thetaDot2 + thetaDot2**2)
    C2 = m2*L1*a2*np.sin(theta2)*(thetaDot1**2)
    C = np.array([C1, C2])
    
    #Gravity
    G1 = (m1*a1 + m2*L1)*gravity*np.cos(theta1) + m2*a2*gravity*np.cos(theta1 + theta2)
    G2 = m2*a2*gravity*np.cos(theta1 + theta2)
    G = np.array([G1, G2])
    
    #get error in position and velocity at current time index 
    PositionError = np.array([theta1Desired[i] - theta1, theta2Desired[i] - theta2])
    VelocityError = np.array([thetaDot1Desired[i] - thetaDot1, thetaDot2Desired[i] - thetaDot2])
    
    #tau gravity compensation and pd control
    tau = Kp * PositionError + Kd * VelocityError + G
    
    #solve for acceleration of each joint
    thetaDoubleDot = np.linalg.solve(M, tau - C - G)
    
    #return derivatives of the state variables
    return [thetaDot1, thetaDot2, thetaDoubleDot[0], thetaDoubleDot[1]]


#function to animate the robot 
def animate(event):
    
    #global flag updated here
    global animationRunning

    if event.key == 'enter' and animationRunning == False:

        #set animationRan flag so animation only runs once
        figure.set_title(f'Animating {selectedShape}')
        animationRunning = True

        #store EE trace for visualization
        eeTraceX = []
        eeTraceY = []

        #animate the trajectory from the ode simulation
        for i in range(len(statevarlist)):
            x1, y1, x2, y2 = forwardKinematics(theta1[i], theta2[i])
            
            link1.set_data([0 , x1], [0, y1])
            link2.set_data([x1, x2], [y1,  y2])
            
            #trace the end effector path
            eeTraceX.append(x2)
            eeTraceY.append(y2)
            eePath.set_data(eeTraceX, eeTraceY)

            plt.draw()
            plt.pause(tStep)
        
        #set animation running flag back to false and update title
        animationRunning = False
        figure.set_title(f'Press Enter to animate {selectedShape}')
        
        #clear the trace for next animation
        eePath.set_data([], [])
        plt.draw()


#************************ MAIN ***************************
#prompt user for shape selection
print("\n Cartesian Trajectory Shape Selection")
print("1. Oval")
print("2. Square")
print("3. Triangle")
shapeChoice = input("Enter shape (1/2/3): ").strip()

#map user input to shape name
shapeMap = {'1': 'oval', '2': 'square', '3': 'triangle'}
selectedShape = shapeMap.get(shapeChoice, 'oval')
print(f"Selected shape: {selectedShape}")

#generate waypoints for selected shape
waypoints, isContinuous = generateShapeWaypoints(selectedShape, shapeCenter, shapeSize)

#create figure object for plotting
fig, figure = plt.subplots(figsize=(7, 7))

#add minimum and maximum workspace circles to show annulus (area between inner and outer circles)
innerCircle = Circle((0, 0), minWS, fill=False, edgecolor='red', linewidth=1)
outerCircle = Circle((0, 0), maxWS, fill=False, edgecolor='blue', linewidth=1)
figure.add_patch(innerCircle)
figure.add_patch(outerCircle)
figure.set_xlim(-maxWS - 0.2, maxWS + 0.2)
figure.set_ylim(-maxWS - 0.2, maxWS + 0.2)
figure.set_xlabel('X (meters)')
figure.set_ylabel('Y (meters)')

#plot the target shape path
waypointsArray = np.array(waypoints)
shapePath, = figure.plot(waypointsArray[:, 0], waypointsArray[:, 1], 'g--', linewidth=2, label=f'Target {selectedShape}')

#show robot at starting configuration (first waypoint)
startTheta1, startTheta2 = inverseKinematics(waypoints[0][0], waypoints[0][1])
x1Start, y1Start, x2Start, y2Start = forwardKinematics(startTheta1, startTheta2)
link1, = figure.plot([0, x1Start], [0, y1Start], 'o-', linewidth=3, color='orange')
link2, = figure.plot([x1Start, x2Start], [y1Start, y2Start], 'o-', linewidth=3, color='orange')

#add line for tracing end effector path during animation
eePath, = figure.plot([], [], 'r-', linewidth=1.5, alpha=0.7, label='EE Path')

#add legend
figure.legend(loc='upper right')

#generate full cartesian trajectory through all waypoints
if isContinuous:
    #use continuous trajectory for smooth shapes like oval
    ovalTime = 8.0  #total time to complete the oval (seconds)
    fullTrajectory = generateContinuousTrajectory(waypoints, ovalTime)
else:
    #use piecewise trajectory with pauses for shapes with corners
    fullTrajectory = generateFullTrajectory(waypoints)

#convert to array and extract desired states
desiredTrajectory = np.array(fullTrajectory)
theta1Desired = desiredTrajectory[:, 0]
theta2Desired = desiredTrajectory[:, 1]
thetaDot1Desired = desiredTrajectory[:, 2]
thetaDot2Desired = desiredTrajectory[:, 3]

#create time array to map to desired trajectory
tSpan = np.linspace(0, len(desiredTrajectory) * tStep, len(desiredTrajectory))

#initial state (starting at first waypoint)
initialState = [startTheta1, startTheta2, 0, 0]  

#run ODE simulation to compute the actual trajectory
statevarlist = odeint(manipulatorEOM, initialState, tSpan, args=(theta1Desired, theta2Desired, thetaDot1Desired, thetaDot2Desired, tSpan, Kp, Kd))

#actual joint angles
theta1 = statevarlist[:, 0]  
theta2 = statevarlist[:, 1] 
thetaDot1 = statevarlist[:, 2]
thetaDot2 = statevarlist[:, 3]


#************************ PLOTS  ***************************
#convert error and theta to degrees for plotting
theta1DesiredDeg = np.rad2deg(theta1Desired)
theta1ActualDeg = np.rad2deg(theta1)
theta2DesiredDeg = np.rad2deg(theta2Desired)
theta2ActualDeg = np.rad2deg(theta2)
theta1Error = np.rad2deg(theta1Desired - theta1)
theta2Error = np.rad2deg(theta2Desired - theta2)

#compute cartesian tracking error
xDesired = []
yDesired = []
xActual = []
yActual = []
for i in range(len(theta1)):
    _, _, xd, yd = forwardKinematics(theta1Desired[i], theta2Desired[i])
    _, _, xa, ya = forwardKinematics(theta1[i], theta2[i])
    xDesired.append(xd)
    yDesired.append(yd)
    xActual.append(xa)
    yActual.append(ya)
xDesired = np.array(xDesired)
yDesired = np.array(yDesired)
xActual = np.array(xActual)
yActual = np.array(yActual)
cartesianError = np.sqrt((xDesired - xActual)**2 + (yDesired - yActual)**2) * 1000  #in mm

#plot desired vs actual and error for each joint 
plt.figure(figsize=(12, 10))
plt.suptitle(f'Controller Performance - {selectedShape.capitalize()} Trajectory', fontsize=14)

plt.subplot(2, 2, 1)
plt.plot(tSpan, theta1DesiredDeg, 'b--', linewidth=2, label='Desired')
plt.plot(tSpan, theta1ActualDeg, 'r-', linewidth=1, label='Actual')
plt.ylabel('Joint 1 Angle (deg)')
plt.title('Joint 1 Position Tracking')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(tSpan, theta2DesiredDeg, 'b--', linewidth=2, label='Desired')
plt.plot(tSpan, theta2ActualDeg, 'r-', linewidth=1, label='Actual')
plt.ylabel('Joint 2 Angle (deg)')
plt.title('Joint 2 Position Tracking')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(tSpan, theta1Error, 'k-', linewidth=1.5, label='Joint 1')
plt.plot(tSpan, theta2Error, 'm-', linewidth=1.5, label='Joint 2')
plt.ylabel('Joint Error (deg)')
plt.xlabel('Time (s)')
plt.title('Joint Tracking Errors')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(tSpan, cartesianError, 'g-', linewidth=1.5)
plt.ylabel('Cartesian Error (mm)')
plt.xlabel('Time (s)')
plt.title('End-Effector Cartesian Tracking Error')
plt.grid(True)

plt.tight_layout()

#set title for animation figure
figure.set_title(f'Press Enter to animate {selectedShape}')

#connect event handler to start animation on Enter key
fig.canvas.mpl_connect('key_press_event', animate)

plt.show()
