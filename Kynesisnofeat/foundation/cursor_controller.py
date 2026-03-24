import pyautogui

pyautogui.FAILSAFE = False

class CursorController:
    def __init__(self, smoothing=5):
        self.smoothing = smoothing
        self.prev_x, self.prev_y = 0, 0
    
    def move_to(self, x, y):
        curr_x = self.prev_x + (x - self.prev_x) / self.smoothing
        curr_y = self.prev_y + (y - self.prev_y) / self.smoothing
        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y

    def click(self):
        pyautogui.click()
        
    def scroll(self, dy, speed=2.0):
        pyautogui.scroll(int(dy * speed))
