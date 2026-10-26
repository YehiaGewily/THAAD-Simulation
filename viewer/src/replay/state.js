// Playback state shared by the controls, the camera director and the frame loop.
import * as THREE from 'three';
import { tStart, tEnd } from './data.js';

export const SPEEDS = [0.25, 1, 2, 5, 10, 20];

export const state = {
  speed: 5,
  simTime: tStart,
  playing: true,
  scrubbing: false,
  camMode: 'cinematic',
  // Free orbit can follow a moving vehicle, so you can spin around it while it flies
  orbitFollow: null,        // 'threat' | 'kv' | null
  followFresh: false,
  chaseDist: 0.5,
  snapCamera: true,
  layers: { radar: true, track: true, cov: true, los: true, labels: true, slowmo: true },
  sciStyle: true,
  analysisOpen: false,
  analysisTab: 'kin',
  bdaDismissed: false,
  bdaShown: false,
};

export const realClock = { t: 0 };
export const camLookTarget = new THREE.Vector3();

// Optional deep link, e.g. replay.html#t=57&cam=threat&speed=1&paused&focus
export const hashArgs = new URLSearchParams(location.hash.slice(1));

export function seek(t) { state.simTime = THREE.MathUtils.clamp(t, tStart, tEnd); state.bdaDismissed = false; }
