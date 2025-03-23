import React, { useEffect, useRef, useState } from 'react';

// Advanced animated noise with JavaScript-controlled seed
export function AnimatedNoise({
  baseFrequency = '0.65',
  numOctaves = '3',
  seedInterval = 200, // Time in ms between seed changes
  ...props
}) {
  const [seed, setSeed] = useState(0);
  const requestRef = useRef();
  const previousTimeRef = useRef();
  const filterId = `jsNoiseFilter_${Math.random().toString(36).substr(2, 9)}`;

  // Animation loop to update the seed
  const animate = (time) => {
    if (previousTimeRef.current === undefined) {
      previousTimeRef.current = time;
    }

    const elapsed = time - previousTimeRef.current;

    if (elapsed > seedInterval) {
      setSeed(prevSeed => (prevSeed + 1) % 100); // Cycle through 0-99
      previousTimeRef.current = time;
    }

    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [seedInterval]); // Re-create effect if interval changes

  return (
    <svg
      viewBox="0 0 200 200"
      xmlns="http://www.w3.org/2000/svg"
    //   style={{ width: '100%', height: '100%' }}
      {...props}
    >
      <filter id={filterId}>
        <feTurbulence
          type="fractalNoise"
          baseFrequency={baseFrequency}
          numOctaves={numOctaves}
          seed={seed}
          stitchTiles="stitch"
        />
        <feBlend mode="screen"/>
      </filter>

      <rect width="500" height="500" opacity="0.5" filter={`url(#${filterId})`} />
    </svg>
  );
}

// More advanced noise with controllable parameters
export function AdvancedAnimatedNoise({
  baseFrequencyX = 0.65,
  baseFrequencyY = 0.65,
  numOctaves = 3,
  seedInterval = 150,
  frequencyInterval = 5000, // Time in ms between frequency changes
  ...props
}) {
  const [seed, setSeed] = useState(0);
  const [frequency, setFrequency] = useState({ x: baseFrequencyX, y: baseFrequencyY });
  const seedTimerRef = useRef();
  const freqTimerRef = useRef();
  const filterId = `advancedNoiseFilter_${Math.random().toString(36).substr(2, 9)}`;

  // Setup seed animation
  useEffect(() => {
    seedTimerRef.current = setInterval(() => {
      setSeed(prevSeed => (prevSeed + 1) % 100);
    }, seedInterval);

    return () => clearInterval(seedTimerRef.current);
  }, [seedInterval]);

  // Setup frequency animation
  useEffect(() => {
    let direction = 1;
    freqTimerRef.current = setInterval(() => {
      setFrequency(prev => {
        // Change direction when reaching limits
        if (prev.x > baseFrequencyX * 1.3 || prev.x < baseFrequencyX * 0.7) {
          direction *= -1;
        }

        const newValue = prev.x + (0.01 * direction);
        return { x: newValue, y: newValue };
      });
    }, frequencyInterval / 50); // Divide for smoother transitions

    return () => clearInterval(freqTimerRef.current);
  }, [baseFrequencyX, baseFrequencyY, frequencyInterval]);

  return (
    <svg
      viewBox="0 0 200 200"
      xmlns="http://www.w3.org/2000/svg"
      style={{ width: '100%', height: '100%' }}
      {...props}
    >
      <filter id={filterId}>
        <feTurbulence
          type="fractalNoise"
          baseFrequency={`${frequency.x} ${frequency.y}`}
          numOctaves={numOctaves}
          seed={seed}
          stitchTiles="stitch"
        />
      </filter>

      <rect width="100%" height="100%" filter={`url(#${filterId})`} />
    </svg>
  );
}

