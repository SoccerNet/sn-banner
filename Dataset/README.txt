This dataset has been converted to be used by mmseg, without any splits

 
Labels are stored as images (png or bmp) with a specific color associated to each pixel. The color code for is given below.

                             R-G-B
---------------------------------------------------------
Outside billboards

    1.                    000-000-000  (black)

---------------------------------------------------------
Inside billboards

    2. Billboard          255-255-255  (white)

    3. Field player       255-000-000  (red)
    4. Goalkeeper         000-255-000  (green)
    5. Referee            000-000-255  (blue)
    6. Assistant referee  255-255-000  (yellow)
    7. Other human        255-000-255  (pink)

    8. Ball               000-255-255  (turquoise)

    9. Goal post          128-000-000  (dark red)
   10. Goal net           000-128-000  (dark green)
   12. Net post           000-000-128  (dark blue)
   13. Cross-bar          064-064-064  (dark gray)

   14. Corner flag        128-128-000  (dark yellow)
   15. Assistant flag     128-000-128  (purple)

   16. Microphone         000-128-128  (dark turquoise)
   17. Camera             255-128-000  (orange)
   
   18. Other object       192-192-192  (light gray)
  
   19. Don't care         128-128-128  (gray)
---------------------------------------------------------
