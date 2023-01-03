# Interactive Bode Plot
Small interactive tool for visualization of the effects of transfer function poles and zeros in the complex plane on the Bode Plot. Uses pygame for GUI interface.


## Functionality
Place poles $p_k$ (red crosses) by leftclicking and zeros $z_k$ (blue circles) by rightclicking in the complex plane on the left. "R" resets the poles and zeros. The right side shows the bode plot (magnitude in dB and phase in deg) of the resulting system transfer function

$$ H(j\omega) = \frac{(j\omega - z_0)(j\omega - z_1)\cdots(j\omega - z_m)}{(j\omega - p_0)(j\omega - p_1)\cdots(j\omega - p_n)} $$



![bode](https://user-images.githubusercontent.com/105657697/210376900-c4fba7df-72c5-47a4-bd38-634ba15579ce.PNG)


## Demo

https://user-images.githubusercontent.com/105657697/210376957-3141c4a0-306e-4387-a82b-37e00dc49851.mp4

