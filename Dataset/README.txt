This dataset has been converted to be used by mmseg, without any splits

 
Labels are stored as images (png or bmp) with a specific color associated to each pixel. The color code for is given below.

                             R-G-B
---------------------------------------------------------
Outside billboards

    0.                    000-000-000  (black)

---------------------------------------------------------
Inside billboards

    1. Billboard          255-255-255  (white)

    2. Field player       255-000-000  (red)
    3. Goalkeeper         000-255-000  (green)
    4. Referee            000-000-255  (blue)
    5. Assistant referee  255-255-000  (yellow)
    6. Other human        255-000-255  (pink)

    7. Ball               000-255-255  (turquoise)

    8. Goal post          128-000-000  (dark red)
    9. Goal net           000-128-000  (dark green)
   10. Net post           000-000-128  (dark blue)
   11. Cross-bar          064-064-064  (dark gray)

   12. Corner flag        128-128-000  (dark yellow)
   13. Assistant flag     128-000-128  (purple)

   14. Microphone         000-128-128  (dark turquoise)
   15. Camera             255-128-000  (orange)
   
   16. Other object       192-192-192  (light gray)
  
   17. Don't care         128-128-128  (gray)
---------------------------------------------------------
