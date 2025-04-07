from math import sqrt
from turtle import settiltangle
import cv2
import mediapipe as mp
import numpy as np
import time
import random

# === Load and check image ===
def load_and_check_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image file not found: {path}")
    return img

# === Load Assets ===
rock_img = load_and_check_image("rock.png")
paper_img = load_and_check_image("paper.png")
scissors_img = load_and_check_image("scissors.png")
shoot_img = load_and_check_image("shoot.png")
win_img = load_and_check_image("you_win.png")
lose_img = load_and_check_image("you_lose.png")
draw_img = load_and_check_image("draw.png")
explosion_img = load_and_check_image("explosion.png")

# === Helpers ===
def resize_image(img, width):
    height = int(img.shape[0] * (width / img.shape[1]))
    return cv2.resize(img, (width, height))

def overlay_with_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    if ow == 0 or oh == 0 or x + ow > bw or y + oh > bh or x < 0 or y < 0:
        return background

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+oh, x:x+ow, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * background[y:y+oh, x:x+ow, c]
            )
    else:
        background[y:y+oh, x:x+ow] = overlay
    return background

def overlay_text(frame, text, position, font_scale=2, color=(0, 0, 255), thickness=3):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# === Draw Menu ===
def draw_menu(frame):
    # Fill the entire frame with a solid color
    menu_bg_color = (200, 128, 128)  
    frame[:] = menu_bg_color

    # Draw the title at the top
    overlay_text(frame, "Rock Paper Scissors", (50, 100), font_scale=1, color=(255, 255, 255), thickness=3)

    # Define button dimensions and positions
    btn_w, btn_h = 300, 80
    start_btn_pos = (50, 150)     # Top-left corner of Start button
    settings_btn_pos = (50, 250)  # Top-left corner of Settings button
    skeleton_btn_pos = (50, 350)  # Top-left corner of Skelton button

    # Draw Start button
    cv2.rectangle(frame, start_btn_pos, (start_btn_pos[0] + btn_w, start_btn_pos[1] + btn_h), (0, 255, 0), -1)
    overlay_text(frame, "START", (start_btn_pos[0] + 50, start_btn_pos[1] + 55), font_scale=1.5, color=(255, 255, 255), thickness=2)

    # Draw Settings button
    cv2.rectangle(frame, settings_btn_pos, (settings_btn_pos[0] + btn_w, settings_btn_pos[1] + btn_h), (128, 128, 128), -1)
    overlay_text(frame, "SETTINGS", (settings_btn_pos[0] + 20, settings_btn_pos[1] + 55), font_scale=1.5, color=(255, 255, 255), thickness=2)

    if settings_active == True:
     # Draw Skeleton settings button
     cv2.rectangle(frame, skeleton_btn_pos, (skeleton_btn_pos[0] + 500, skeleton_btn_pos[1] + btn_h), (128, 128, 128), -1)
     overlay_text(frame, "ENABLE/DISABLE SKELETON", (skeleton_btn_pos[0] + 20, skeleton_btn_pos[1] + 55), font_scale=1, color=(255, 255, 255), thickness=2)

# === Check if mouse click is inside a rectangle ===
def is_click_inside(x, y, top_left, width, height):
    tx, ty = top_left
    return tx <= x <= tx + width and ty <= y <= ty + height

# Global variable for menu state
menu_active = True

# Mouse callback for menu screen
def menu_mouse_callback(event, x, y, flags, param):
    global menu_active
    if event == cv2.EVENT_LBUTTONDOWN:
        # Define button dimensions matching draw_menu()
        btn_w, btn_h = 300, 80
        start_btn_pos = (50, 150)
        settings_btn_pos = (50, 250)
        if is_click_inside(x, y, start_btn_pos, btn_w, btn_h):
            menu_active = False
        # Settings button doesn't do anything yet.
        
# ===Gesture Classification ===
def get_hand_gesture(hand_landmarks, h, w):
    """
    Determines a simple gesture (rock, paper, or scissors)
    by checking whether each finger (ignoring the thumb) is extended.
    
    For each finger (index, middle, ring, pinky), we compare the y-coordinate 
    of the tip with the y-coordinate of the PIP joint.
    
    In the image coordinate system, a lower y value means higher on the image.
    If a finger is extended, its tip should be above (i.e. have a smaller y value than) the PIP joint.
    """
    #get the list of landmarks.
    lm = hand_landmarks.landmark

    # Function to determine if a finger is extended.
    # Measures the distance between the knuckle and the finger tip
    # Uses the distance from the wrist to base of the thumb as a threshold distance to classify a finger  as extended
    def is_finger_extended(tip_id, mcp_id):
        tip_y = lm[tip_id].y * h
        mcp_y = lm[mcp_id].y * h
        tip_x = lm[tip_id].x * w
        mcp_x = lm[mcp_id].x * w
        
        # Calculate thumb CMC to wrist distance as reference
        wrist_x = lm[0].x * w
        wrist_y = lm[0].y * h
        thumb_cmc_x = lm[1].x * w
        thumb_cmc_y = lm[1].y * h
        hand_size = sqrt((wrist_x - thumb_cmc_x)**2 + (wrist_y - thumb_cmc_y)**2)

        distance = 1
        #check that the coordinates are not null before attempting to measure the distance
        if tip_y is not None and tip_x is not None and mcp_y is not None and mcp_x is not None:
            distance = round(sqrt((mcp_x - tip_x)**2 + (mcp_y - tip_y)**2))
          
        threshold = round(max(50, hand_size * 0.4))

        if distance > threshold:
            return True
        else:
            return False

    index = is_finger_extended(8, 5)
    middle = is_finger_extended(12, 9)
    ring = is_finger_extended(16, 13)
    pinky = is_finger_extended(20, 17)
    count = sum([index, middle, ring, pinky])

    if count == 0:
        return "rock"
    elif count == 4:
        return "paper"
    elif count == 2 and index and middle and not ring and not pinky:
        return "scissors"
    else:
        return "unknown"

def draw_hand_skeleton(frame, hand_landmarks, gesture):
    # Draw hand landmarks and connections
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())
    
    # Draw gesture label near wrist
    wrist = hand_landmarks.landmark[0]
    h_frame, w_frame = frame.shape[:2]
    text_position = (int(wrist.x * w_frame) - 50, int(wrist.y * h_frame) - 50)
    overlay_text(frame, gesture.upper(), text_position, 1, (0, 255, 255), 2)

def animate_arm_countdown(cap, computer_choice, hands):
    fist_down = load_and_check_image("fistmovingdown.png")
    fist_up = load_and_check_image("fistup.png")
    fist_up_move = load_and_check_image("fistmovingup.png")
    shoot_overlay = load_and_check_image("shoot.png")  

    bottom_poses = {
        "rock": load_and_check_image("fistrock.png"),
        "paper": load_and_check_image("fistpaper.png"),
        "scissors": load_and_check_image("fistscissors.png"),
        "shoot_rock": load_and_check_image("fistrock.png"),
        "shoot_paper": load_and_check_image("comppaper.png"),
        "shoot_scissors": load_and_check_image("compscissors.png"),
    }
    sequence = ["rock", "paper", "scissors"]
    pose_duration = 0.8  # seconds to show each pose
    
    for move in sequence:
        start_time = time.time()
        
        # Animate the arm movement
        for frame_img in [fist_up_move, fist_up, fist_down]:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            
            # Process hand in real-time during movement
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Overlay animation frame
            anim_frame = overlay_with_alpha(frame.copy(), 
                                         resize_image(frame_img, 350),
                                         (frame.shape[1] - 350)//2,
                                         (frame.shape[0] - 350)//2)
            
            # Draw hand skeleton if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = get_hand_gesture(hand_landmarks, *frame.shape[:2])
                    if skeleton_active == True:
                      draw_hand_skeleton(anim_frame, hand_landmarks, gesture)
            
            cv2.imshow("Rock Paper Scissors", anim_frame)
            cv2.waitKey(50)
        
        # Show the pose with continuous camera feed
        while time.time() - start_time < pose_duration:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            # Process hand in real-time
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Overlay pose image
            pose_frame = overlay_with_alpha(frame.copy(),
                                          resize_image(bottom_poses[move], 350),
                                          (frame.shape[1] - 350)//2,
                                          (frame.shape[0] - 350)//2)
            
            # Draw hand skeleton if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = get_hand_gesture(hand_landmarks, *frame.shape[:2])
                    if skeleton_active == True:
                     draw_hand_skeleton(pose_frame, hand_landmarks, gesture)
            
            # Add countdown text
            overlay_text(pose_frame, move.upper(), 
                        (frame.shape[1]//2 - 50, 50), 
                        2, (0, 255, 255), 3)
            
            cv2.imshow("Rock Paper Scissors", pose_frame)
            cv2.waitKey(1)

    # === Final "Shoot!" Frame ===
    shoot_start = time.time()
    best_gesture = "unknown"
    max_hand_size = 0

    # Calculate positions
    comp_img = resize_image(bottom_poses[f"shoot_{computer_choice}"], 350)
    frame_h, frame_w = frame.shape[:2] if 'frame' in locals() else (480, 640)
    cx = (frame_w - comp_img.shape[1]) // 2  # Computer hand X position
    cy = (frame_h - comp_img.shape[0]) // 2  # Computer hand Y position
    sx = (frame_w - shoot_overlay.shape[1]) // 2  # Shoot overlay X
    sy = (frame_h - shoot_overlay.shape[0]) // 2 - 130  # Shoot overlay Y

    while time.time() - shoot_start < 2:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        # Process frame for hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate hand size (wrist to middle finger MCP)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                hand_size = sqrt((wrist.x-middle_mcp.x)**2 + (wrist.y-middle_mcp.y)**2)
                
                # Only keep the largest hand
                if hand_size > max_hand_size:
                    max_hand_size = hand_size
                    best_gesture = get_hand_gesture(hand_landmarks, frame.shape[0], frame.shape[1])
        
        # Show the frame with overlays
        frame = overlay_with_alpha(frame, shoot_overlay, sx, sy)
        frame = overlay_with_alpha(frame, comp_img, cx, cy)
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(1)

    return best_gesture

def show_ready_steady_go(cap, hands):
    messages = ["Ready...", "Steady...", "GO!"]
    durations = [1000, 1000, 800]  # in milliseconds

    for i, msg in enumerate(messages):
        start_time = time.time()
        while time.time() - start_time < durations[i]/1000:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            # Process hands in real-time during countdown
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            overlay_text(frame, msg, (150, 250), 2.5, (0, 255, 0), 5)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = get_hand_gesture(hand_landmarks, frame.shape[0], frame.shape[1])
                    if skeleton_active == True:
                      draw_hand_skeleton(frame, hand_landmarks, current_gesture)
            
            cv2.imshow("Rock Paper Scissors", frame)
            cv2.waitKey(1)

def animate_battle(frame, player_img, computer_img, outcome_img, winner="player"):
    explosion_img = load_and_check_image("explosion.png")
    explosion_img = resize_image(explosion_img, 300)

    frame_height, frame_width = frame.shape[:2]
    player_img = resize_image(player_img, 200)
    computer_img = resize_image(computer_img, 200)

    py = cy = frame_height // 2 - player_img.shape[0] // 2
    px_start = 0
    cx_start = frame_width - computer_img.shape[1]
    center_x = (frame_width - player_img.shape[1]) // 2

    # === 0. Dramatic pause with zoom + shake ===
    pause_frames = 20
    for i in range(pause_frames):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.flip(f, 1)

        scale = 1.0 + 0.03 * (i / pause_frames)
        shake = 4

        p_scaled = resize_image(player_img, int(200 * scale))
        c_scaled = resize_image(computer_img, int(200 * scale))

        px = px_start + np.random.randint(-shake, shake)
        py_shake = py + np.random.randint(-shake, shake)
        cx = cx_start + np.random.randint(-shake, shake)
        cy_shake = cy + np.random.randint(-shake, shake)

        px += (200 - p_scaled.shape[1]) // 2
        py_shake += (200 - p_scaled.shape[0]) // 2
        cx += (200 - c_scaled.shape[1]) // 2
        cy_shake += (200 - c_scaled.shape[0]) // 2

        f = overlay_with_alpha(f, p_scaled, px, py_shake)
        f = overlay_with_alpha(f, c_scaled, cx, cy_shake)

        cv2.imshow("Rock Paper Scissors", f)
        cv2.waitKey(30)

    # === 1. Dash into center ===
    steps = 10
    for i in range(steps):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.flip(f, 1)
        
        t = i / steps
        px = int(px_start + t * (center_x - px_start))
        cx = int(cx_start - t * (cx_start - center_x))
        f = overlay_with_alpha(f, player_img, px, py)
        f = overlay_with_alpha(f, computer_img, cx, cy)
        cv2.imshow("Rock Paper Scissors", f)
        cv2.waitKey(20)

    # === 2. Explosion ===
    ret, f = cap.read()
    if not ret:
        return
    f = cv2.flip(f, 1)
    
    fx = (frame_width - explosion_img.shape[1]) // 2
    fy = (frame_height - explosion_img.shape[0]) // 2
    f = overlay_with_alpha(f, explosion_img, fx, fy)
    cv2.imshow("Rock Paper Scissors", f)
    cv2.waitKey(500)

    # === 3. Knockback with spin ===
    loser_img = computer_img if winner == "player" else player_img
    winner_img = player_img if winner == "player" else computer_img

    loser_x = center_x
    loser_y = py
    winner_x = center_x
    winner_y = py

    for i in range(25):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.flip(f, 1)
        
        angle = i * 25
        scale = 1.0 - (i * 0.02)
        M = cv2.getRotationMatrix2D((loser_img.shape[1] // 2, loser_img.shape[0] // 2), angle, scale)
        rotated = cv2.warpAffine(loser_img, M, (loser_img.shape[1], loser_img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        move_x = loser_x + (i * 25 if winner == "player" else -i * 25)
        if 0 <= move_x <= frame_width - rotated.shape[1]:
            f = overlay_with_alpha(f, rotated, move_x, loser_y)
        f = overlay_with_alpha(f, winner_img, winner_x, winner_y)

        cv2.imshow("Rock Paper Scissors", f)
        cv2.waitKey(20)

    # === 4. Winner zoom-in linger ===
    for i in range(20):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.flip(f, 1)
        
        zoom = 1.0 + (i * 0.02)
        winner_zoom = resize_image(winner_img, int(200 * zoom))
        zoom_x = (frame_width - winner_zoom.shape[1]) // 2
        zoom_y = (frame_height - winner_zoom.shape[0]) // 2
        f = overlay_with_alpha(f, winner_zoom, zoom_x, zoom_y)
        cv2.imshow("Rock Paper Scissors", f)
        cv2.waitKey(50)

    # === 5. Final result image ===
    ret, final = cap.read()
    if not ret:
        return
    final = cv2.flip(final, 1)
    
    outcome = resize_image(outcome_img, 300)
    ox = (frame_width - outcome.shape[1]) // 2
    oy = (frame_height - outcome.shape[0]) // 2
    final = overlay_with_alpha(final, outcome, ox, oy)
    cv2.imshow("Rock Paper Scissors", final)
    cv2.waitKey(2000)

# === Game Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)
game_active = False

# Set up window for menu clicks
cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)
# Register mouse callback for the menu
def menu_mouse_callback(event, x, y, flags, param):
    global menu_active, settings_active, skeleton_active
    if event == cv2.EVENT_LBUTTONDOWN:
        # Button dimensions as defined in draw_menu()
        btn_w, btn_h = 300, 80
        start_btn_pos = (50, 150)
        settings_btn_pos = (50, 250)
        skeleton_btn_pos = (50, 350)
        if start_btn_pos[0] <= x <= start_btn_pos[0] + btn_w and start_btn_pos[1] <= y <= start_btn_pos[1] + btn_h:
            # Start button clicked: exit menu and go to game stage
            global menu_active
            menu_active = False

        if settings_btn_pos[0] <= x <= settings_btn_pos[0] + btn_w and settings_btn_pos[1] <= y <= settings_btn_pos[1] + btn_h:
            # Settings clicked: show game settings
            global settings_active
            settings_active = True

        if skeleton_btn_pos[0] <= x <= skeleton_btn_pos[0] + btn_w and skeleton_btn_pos[1] <= y <= skeleton_btn_pos[1] + btn_h:
            # Skeleton button clicked: Toggle skeleton
            global skeleton_active            
            if skeleton_active == True:
                skeleton_active = False
            else:
                skeleton_active = True

cv2.setMouseCallback("Rock Paper Scissors", menu_mouse_callback)

# Initially, show the menu until Start is clicked.
menu_active = True
settings_active = False
skeleton_active = True
while menu_active:
    ret, menu_frame = cap.read()
    if not ret:
        break
    menu_frame = cv2.flip(menu_frame, 1)
    draw_menu(menu_frame)
    cv2.imshow("Rock Paper Scissors", menu_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        menu_active = False
        settings_active = False
        game_active = True  

# === Main Game Loop ===
while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    #show hand skeleton and gesture when hand is detected
    current_gesture = "No Hand"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            current_gesture = get_hand_gesture(hand_landmarks, h, w)
            if skeleton_active == True:
             draw_hand_skeleton(frame, hand_landmarks, current_gesture)

    if not game_active:
        overlay_text(frame, "Press 'g' to start game", (10, 40), 1, (255, 255, 0), 2)
        overlay_text(frame, f"Current: {current_gesture}", (10, 80), 1, (255, 255, 255), 2)

    cv2.imshow("Rock Paper Scissors", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        game_active = True
        
        # Wait for fist in center
        steady_start = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = get_hand_gesture(hand_landmarks, frame.shape[0], frame.shape[1])
                    if skeleton_active == True:
                       draw_hand_skeleton(frame, hand_landmarks, current_gesture)
                    #Fist = rock, centre = 0,0
                    if gesture == "rock":
                        wrist = hand_landmarks.landmark[0]
                        if 0.3 < wrist.x < 0.7 and 0.3 < wrist.y < 0.7:
                            if steady_start is None:
                                steady_start = time.time()
                            elif time.time() - steady_start > 1.0:
                                break
                        else:
                            steady_start = None
            
            overlay_text(frame, "Hold your fist in the center", (50, 50), 1, (0, 255, 0), 2)
            cv2.imshow("Rock Paper Scissors", frame)
            cv2.waitKey(1)
            
            if steady_start and time.time() - steady_start > 1.0:
                break

        # Countdown animation
        show_ready_steady_go(cap, hands)
        
        # Play the game
        computer_choice = random.choice(["rock", "paper", "scissors"])
        player_gesture = animate_arm_countdown(cap, computer_choice, hands)
        
        if player_gesture not in ["rock", "paper", "scissors"]:
            print("Invalid gesture detected")
            game_active = False
            continue
        
        #Determine outcome
        if player_gesture == computer_choice:
            outcome_img = draw_img
            winner = "none"
        elif (player_gesture == "rock" and computer_choice == "scissors") or \
             (player_gesture == "paper" and computer_choice == "rock") or \
             (player_gesture == "scissors" and computer_choice == "paper"):
            outcome_img = win_img
            winner = "player"
        else:
            outcome_img = lose_img
            winner = "computer"
        
        # Show battle animation
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            player_img = {"rock": rock_img, "paper": paper_img, "scissors": scissors_img}[player_gesture]
            comp_img = {"rock": rock_img, "paper": paper_img, "scissors": scissors_img}[computer_choice]
            animate_battle(frame, player_img, comp_img, outcome_img, winner)
        
        game_active = False

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
