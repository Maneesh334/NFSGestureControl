import cv2
import mediapipe as mp
import pydirectinput as pg
import collections
import time

# Keys
KEY_ACCELERATE = 'w'
KEY_BRAKE = 's'
KEY_LEFT = 'a'
KEY_RIGHT = 'd'
KEY_SPECIAL = 'e'

# State tracking for acceleration/brake/special/steering
accel_state = False
brake_state = False
special_state = False
steer_left_down = False
steer_right_down = False

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Deque for smoothing wrist positions
wrist_positions = collections.deque(maxlen=5)

def finger_states(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    index_tip = hand_landmarks.landmark[8]
    index_mcp = hand_landmarks.landmark[5]
    middle_tip = hand_landmarks.landmark[12]
    middle_mcp = hand_landmarks.landmark[9]
    ring_tip = hand_landmarks.landmark[16]
    ring_mcp = hand_landmarks.landmark[13]

    # Check if each finger is up
    thumb_up = thumb_tip.x < thumb_ip.x
    index_up = index_tip.y < index_mcp.y
    middle_up = middle_tip.y < middle_mcp.y
    ring_up = ring_tip.y < ring_mcp.y

    return thumb_up, index_up, middle_up, ring_up

def set_accel_brake_mode(mode):
    global accel_state, brake_state

    if mode == 'accelerate':
        if not accel_state:
            pg.keyDown(KEY_ACCELERATE)
            accel_state = True
        if brake_state:
            pg.keyUp(KEY_BRAKE)
            brake_state = False
    elif mode == 'brake':
        if not brake_state:
            pg.keyDown(KEY_BRAKE)
            brake_state = True
        if accel_state:
            pg.keyUp(KEY_ACCELERATE)
            accel_state = False
    else:
        # No accelerate or brake
        if accel_state:
            pg.keyUp(KEY_ACCELERATE)
            accel_state = False
        if brake_state:
            pg.keyUp(KEY_BRAKE)
            brake_state = False

def set_special_mode(mode):
    global special_state
    if mode == 'special':
        if not special_state:
            pg.keyDown(KEY_SPECIAL)
            special_state = True
    else:
        if special_state:
            pg.keyUp(KEY_SPECIAL)
            special_state = False

def set_steering(wrist_x):
    """
    Highly sensitive steering:
    A very narrow neutral zone is defined around the center.
    If the wrist moves even slightly beyond that zone, we hold down the corresponding key.
    """
    global steer_left_down, steer_right_down

    # Define a very narrow neutral zone for high sensitivity
    neutral_left = 0.4 * frame_width
    neutral_right = 0.6 * frame_width

    if wrist_x < neutral_left:
        # Steer left (hold key)
        if not steer_left_down:
            pg.keyDown(KEY_LEFT)
            steer_left_down = True
        if steer_right_down:
            pg.keyUp(KEY_RIGHT)
            steer_right_down = False
    elif wrist_x > neutral_right:
        # Steer right (hold key)
        if not steer_right_down:
            pg.keyDown(KEY_RIGHT)
            steer_right_down = True
        if steer_left_down:
            pg.keyUp(KEY_LEFT)
            steer_left_down = False
    else:
        # Inside neutral zone, no steering
        if steer_left_down:
            pg.keyUp(KEY_LEFT)
            steer_left_down = False
        if steer_right_down:
            pg.keyUp(KEY_RIGHT)
            steer_right_down = False

with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_up, index_up, middle_up, ring_up = finger_states(hand_landmarks)

                # Raw wrist position
                wrist_x = hand_landmarks.landmark[0].x * frame_width
                wrist_positions.append(wrist_x)
                avg_wrist_x = sum(wrist_positions) / len(wrist_positions)

                # Steering logic
                set_steering(avg_wrist_x)

                # Acceleration/Braking logic:
                # If index_up: accelerate (held down)
                # If not index_up but middle_up: brake
                # Else: none
                if index_up:
                    set_accel_brake_mode('accelerate')
                elif middle_up:
                    set_accel_brake_mode('brake')
                else:
                    set_accel_brake_mode('none')

                # Special key logic:
                if index_up and middle_up and ring_up:
                    set_special_mode('special')
                else:
                    set_special_mode('none')

                # Display info
                cv2.putText(frame, f"Wrist X: {int(avg_wrist_x)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # No hand detected
            set_accel_brake_mode('none')
            set_special_mode('none')
            # Return steering to neutral
            if steer_left_down:
                pg.keyUp(KEY_LEFT)
                steer_left_down = False
            if steer_right_down:
                pg.keyUp(KEY_RIGHT)
                steer_right_down = False

        cv2.imshow('Gesture Steering Control', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
