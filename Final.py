from turtle import color
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

#pause time at each waypoint
pauseTime = .30 

#define workspace annulus
minWS = abs(L1 - L2)  
maxWS = L1 + L2     

#maximum velocity of the end effector m/s
maxVelocity = 0.60

#initialize center of shape and radius/side length size
shapeCenter = [0.4, 0.3] 
sideLength = 0.2  

#notes:
#increased kp until there was no steady state error
#started with Kd at 10 for both joints
#increased kd of first joint to prevent overshoot
#best results: q1[150, 200], q2[35, 100]
#controller gains for joints 1 and 2
Kp = np.array([175, 55])  
Kd = np.array([220, 150])    


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
#input: shape (string: 'star', 'square', 'triangle'), center [x, y], size (radius or side length)
#output: list of waypoints [[x1, y1], [x2, y2], ...]
def generateShapeWaypoints(shape, center, size):
    
    waypoints = []
    cx, cy = center
    
    if shape == 'star':
        #generate 5-pointed star
        outerRadius = size
        innerRadius = size * 0.40  #inner points closer to center
        numPoints = 7
        
        for i in range(numPoints):
            #outer point
            angle = (2 * np.pi * i / numPoints) - np.pi/2  #start at top
            x = cx + outerRadius * np.cos(angle)
            y = cy + outerRadius * np.sin(angle)
            waypoints.append([x, y])
            
            #inner point (between outer points)
            angle = (2 * np.pi * (i + 0.5) / numPoints) - np.pi/2
            x = cx + innerRadius * np.cos(angle)
            y = cy + innerRadius * np.sin(angle)
            waypoints.append([x, y])
        
        #close the star
        waypoints.append(waypoints[0])
            
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
    
    return waypoints

#calculates the duration of the trajectory segment based on the distance between the initial and final positions
#input: initial and final positions [x, y] 
#output: total duration of trajectory (seconds)
def trajectorySegmentDuration(initialPos, finalPos):
    
    #get initial and final x, y coordinates
    initialX, initialY = initialPos
    finalX, finalY = finalPos
    
    #use pythagorean theorem to calculate the distance between the initial and final positions
    distance = np.sqrt((finalX - initialX)**2 + (finalY - initialY)**2)
    
    #total duration is 1.5 times the distance divided by the maximum velocity for cubic trajectory
    totalDuration = 1.5 * distance / maxVelocity
    
    return totalDuration


#takes time within trajectory segment and returns position, velocity, acceleration for one axis
#input: t (seconds), posInitial (meters), posFinal (meters), totalDuration (seconds)
#output: pos (meters), vel (m/s), accel (m/s**2)
def currentTrajectory(t, initialPosition, finalPosition, totalDuration):

    #calculate coefficients for trajectory cubic
    c0 = initialPosition
    c1 = 0
    c2 = 3*(finalPosition - initialPosition)/(totalDuration**2)
    c3 = -2*(finalPosition - initialPosition)/(totalDuration**3)

    #get position, velocity, and acceleration at time t
    position = c0 + c1*t + c2*t**2 + c3*t**3
    velocity = c1 + 2*c2*t + 3*c3*t**2
    
    #acceleration not required. can uncomment later if needed
    #accel = 2*c2 + 6*c3*t

    return position, velocity


#generate straight trajectory segment between two points 
#input: initialPos [x, y], finalPos [x, y] (meters)
#output: trajectory [(theta1, theta2, thetaDot1, thetaDot2)]
def generateTrajectorySegment(initialPosition, finalPosition):
    
    #create empty list to store trajectory
    trajectory = []
    totalDuration = trajectorySegmentDuration(initialPosition, finalPosition)

    t = 0

    #while t is less than the total duration of the trajectory 
    while t < totalDuration:
        
        #get posibion and velocity at time t for x and y
        x, xDot = currentTrajectory(t, initialPosition[0], finalPosition[0], totalDuration)
        y, yDot = currentTrajectory(t, initialPosition[1], finalPosition[1], totalDuration)
        
        #use inverse kinematics to get joint angles at time t
        theta1, theta2 = inverseKinematics(x, y)
        
        #constructjacobian matrix for conversion from cartesian velocity to angular velocity
        J11 = -L1*np.sin(theta1) - L2*np.sin(theta1 + theta2)
        J12 = -L2*np.sin(theta1 + theta2)
        J21 = L1*np.cos(theta1) + L2*np.cos(theta1 + theta2)
        J22 = L2*np.cos(theta1 + theta2)
        J = np.array([[J11, J12], [J21, J22]])
        
        #use jacobian to convert cartesian velocity to angular velocity for each joint
        cartesianVelocity = np.array([xDot, yDot])
        thetaDot = np.linalg.solve(J, cartesianVelocity)
        thetaDot1 = thetaDot[0]
        thetaDot2 = thetaDot[1]
        
        #add to trajectory 
        trajectory.append((theta1, theta2, thetaDot1, thetaDot2))
       
        t += tStep

    #reset time and add pause time at waypoint to assess controller performance. velocity is zero since it's paused
    t = 0
    
    #get joint angles at final position
    finalTheta1, finalTheta2 = inverseKinematics(finalPosition[0], finalPosition[1])
    
    while t < pauseTime:
        #add to trajectory. velocity is zero since it's paused
        trajectory.append((finalTheta1, finalTheta2, 0, 0))
        t += tStep
    
    return trajectory


#Generates complete trajectory between the shape waypoints. Calls generateTrajectorySegment for each segment.
#input: waypoints [[x1, y1], [x2, y2], ...]
#output: planned shape trajectory
def generateFullTrajectory(waypoints):
    
    #initialize emptylist for trajectory
    fullTrajectory = []
    
    for i in range(len(waypoints) - 1):
        
        #generate trajectory between the two consecutive points
        segment = generateTrajectorySegment(waypoints[i], waypoints[i+1])
        
        #add the segment to the full trajectory
        fullTrajectory.extend(segment)
    
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

    #subtract current time from each element in tSpan. Find the minimum, that is current time index
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
        figure.set_title(f'Animating {shapeName} trajectory')
        animationRunning = True

        #store EE trace for animation
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
        figure.set_title(f'Press Enter to animate {shapeName}')
        
        #clear the trace for next animation
        eePath.set_data([], [])
        plt.draw()


#************************ MAIN ***************************

#prompt user to select a shape 
print("\n Cartesian Trajectory Shape Selection")
print("1. Triangle")
print("2. Square")
print("3. Star")

#get input and strip whitespace
shapeChoice = input("Enter shape (1/2/3): ").strip()

#map user input to shape name
shapeOptions = {'1': 'triangle', '2': 'square', '3': 'star'}

#get the shape name, default to triangle if invalid input
shapeName = shapeOptions.get(shapeChoice, 'triangle')
print(f"Selected shape: {shapeName}")

#call generateShapeWaypoints to generate the waypoints for the selected shape
waypoints = generateShapeWaypoints(shapeName, shapeCenter, sideLength)

#create figure object for plotting
fig, figure = plt.subplots(figsize=(7, 7))

#add minimum and maximum workspace circles to show annulus (area between inner and outer circles)
innerCircle = Circle((0, 0), minWS, fill=False, linestyle ='--', edgecolor='black', alpha = .5, linewidth=1)
outerCircle = Circle((0, 0), maxWS, fill=False, linestyle = '--', edgecolor='black', alpha = .5, linewidth=1)
figure.add_patch(innerCircle)
figure.add_patch(outerCircle)
figure.set_xlim(-maxWS - 0.2, maxWS + 0.2)
figure.set_ylim(-maxWS - 0.2, maxWS + 0.2)
figure.set_xlabel('X (meters)')
figure.set_ylabel('Y (meters)')

#add desired shape path to the figure
waypointsArray = np.array(waypoints)
shapePath, = figure.plot(waypointsArray[:, 0], waypointsArray[:, 1], 'k-', linewidth=2, alpha = .90,label= 'Desired Path')
figure.legend(loc='upper left')

#create empty path object to update during animation to trace end effector path
eePath, = figure.plot([], [], 'w-', linewidth=1)

#show robot at starting configuration (first waypoint)
startTheta1, startTheta2 = inverseKinematics(waypoints[0][0], waypoints[0][1])
x1Start, y1Start, x2Start, y2Start = forwardKinematics(startTheta1, startTheta2)
link1, = figure.plot([0, x1Start], [0, y1Start], 'o-', linewidth=3, color='orange')
link2, = figure.plot([x1Start, x2Start], [y1Start, y2Start], 'o-', linewidth=3, color='orange')


#generate full cartesian trajectory through all waypoints using piecewise method
fullTrajectory = generateFullTrajectory(waypoints)

#convert to array and extract desired states
desiredTrajectory = np.array(fullTrajectory)
theta1Desired = desiredTrajectory[:, 0]
theta2Desired = desiredTrajectory[:, 1]
thetaDot1Desired = desiredTrajectory[:, 2]
thetaDot2Desired = desiredTrajectory[:, 3]

#create time array to map to trajectory trajectory
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

#create empty lists for desired and actual x, y values
xDesired = []
yDesired = []
xActual = []
yActual = []

#use inverse kinematics to get desired x, y coordinated at each time step 
for i in range(len(theta1)):
    
    _, _, xD, yD = forwardKinematics(theta1Desired[i], theta2Desired[i])
    _, _, xA, yA = forwardKinematics(theta1[i], theta2[i])
    
    #append values to lists
    xDesired.append(xD)
    yDesired.append(yD)
    xActual.append(xA)
    yActual.append(yA)

#convert lists to arrays
xDesired = np.array(xDesired)
yDesired = np.array(yDesired)
xActual = np.array(xActual)
yActual = np.array(yActual)

#calculate distance error using pythagorean theorem
distanceError = np.sqrt((xDesired - xActual)**2 + (yDesired - yActual)**2) * 1000  #in mm

#plot desired vs actual and error for each joint 
plt.figure(figsize=(12, 10))
plt.suptitle(f'Controller Performance - {shapeName.capitalize()} Trajectory', fontsize=14)

plt.subplot(2, 2, 1)
plt.plot(tSpan, theta1DesiredDeg, 'g-', linewidth=2, label='Desired')
plt.plot(tSpan, theta1ActualDeg, 'r-', linewidth=1, label='Actual')
plt.ylabel('Joint 1 Angle (deg)')
plt.title('Joint 1 Position Tracking')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(tSpan, theta2DesiredDeg, 'g-', linewidth=2, label='Desired')
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
plt.plot(tSpan, distanceError, 'g-', linewidth=1.5)
plt.ylabel('Cartesian Error (mm)')
plt.xlabel('Time (s)')
plt.title('End-Effector Cartesian Tracking Error')
plt.grid(True)

plt.tight_layout()

#set title for animation figure
figure.set_title(f'Press Enter to animate {shapeName} trajectory')

#connect event handler to start animation on Enter key
fig.canvas.mpl_connect('key_press_event', animate)


plt.show()
