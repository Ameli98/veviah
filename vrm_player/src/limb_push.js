// Limb push (2026-10-29, the author: solve the limbs on the body first, then deal with the clothes; on the GPU).
// Mesh vertices inside a limb capsule are moved out to its surface in the vertex shader, after skinning. Capsules
// (vrm.scene.userData.hsinLimbs from colliders.js, skin radius per limb): forearms, hands (stretched to the
// fingertip), thighs, shins. Which vertices a capsule pushes:
//   - never those that hang mostly from the capsule's own limb (they move with it: a sleeve on its forearm, a
//     stocking on its shin);
//   - the skin (measure.skinMesh: the component holding the hands) only by the arms: a hand presses flesh, a leg
//     never pushes the body's own skin (the groin lies inside the thigh capsules at rest);
//   - clothes by every limb (test1's long skirt is skinned to the legs: a raised thigh came through it), minus how
//     deep the vertex already lies inside that capsule in the rest pose (stockings and boots hug the shin closer
//     than a round capsule: only what the motion adds is pushed).
// Any model: nothing is recognised by name. The normals are not changed; the outline pass uses the same vertex code.
// Per frame: updateLimbPush(vrm) after vrm.update.
import * as THREE from 'three';
import { skinMesh } from './measure.js';

const N = 9;                              // left / right forearm, hand, thigh, shin + the bottom
const HAND_TIP = 1.3;                     // hand capsule: wrist .. middle distal joint x this (to the fingertip)
const LIMB_BONES = { LowerArm: 'arm', Hand: 'arm', UpperLeg: 'leg', LowerLeg: 'leg' };
const CODE = { leftarm: 1, rightarm: 2, leftleg: 3, rightleg: 4, hips: 5 };
const LEG_SLOTS = 4;                      // rest depth per leg capsule: attribute hsinRest = (L thigh, L shin, R thigh, R shin)

const DECL = `
uniform vec4 hsinA[${N}];   // xyz = capsule start (world), w = radius (0 = unused)
uniform vec4 hsinB[${N}];   // xyz = capsule end (world), w = limb code (1 / 2 left / right arm, 3 / 4 left / right leg, 5 bottom)
uniform float hsinSlot[${N}];  // leg capsules: index into hsinRest, else -1
uniform vec3 hsinPalm[${N}];   // hand capsules: the way the palm faces (world), else 0
attribute float hsinOwner;  // the limb this vertex hangs from (codes as above, 6 head / neck, 0 none)
attribute float hsinCloth;  // 1 = clothes / hair / anything but the skin component, 2 = such a vertex whose rest normal faces out from the trunk axis, 3 = out and forward (within CLOTH_FRONT_DEG of +z)
attribute vec4 hsinRest;    // clothes: how deep the vertex lies inside each leg capsule in the rest pose
attribute float hsinRestB;  // ... and inside the bottom capsule
uniform vec3 hsinButt[2];   // soft-tissue sway of the left / right buttock (world; jiggle.js), 0 = still
uniform vec3 hsinButtBack;  // the body's backward (world)
uniform mat3 hsinHipsRot;   // the hips' rotation now (world), for hsinButtN
uniform vec4 hsinTrunk;     // the upperChest joint (world) + the trunk's half width: the trunk-cloth rule only where the limb is in front of the trunk, not beside it
uniform vec3 hsinTrunkX;    // the upperChest's sideways axis (world)
uniform vec2 hsinButtShape[2]; // per buttock: lift (m, along the body's up) and bulge (m, outward) from its thigh
uniform vec3 hsinButtUp;    // the body's up (world)
attribute vec3 hsinButtN;   // loose clothes: their outward direction off the bottom at rest (hips frame)
attribute vec2 hsinButtW;   // how much of it this vertex follows (left, right; buttWeights): flesh and what hugs it
attribute vec3 hsinButtL;   // loose clothes over it: the flesh's weights here (left, right) and the rest gap to the skin (m)
`;
const BUTT_PAD_GLSL = '0.03';      // = BUTT_PAD: the tight layer between the skin and loose clothes
const CLOTH_DENT = '0.06';         // m: how far beside a limb the loose cloth it presses back is dented (limb_push, 2026-11-10)
const CLOTH_FRONT_DEG = 60;        // deg from +z: trunk cloth facing forward within this is pressed back by an arm in front of it (the sides are not)
const CLOTH_FRONT_W = 0.6;         // ... and within this fraction of the shoulder half width from the middle: the loose side-front of a shirt pressed by a forearm reaching across to the other arm tore into a black patch (AK-Alpha hand reaction, 2026-11-10)
const CLOTH_MAX = '0.035';         // m: the most a vertex is pressed back by a limb (see the cap below)
const CLOTH_BEHIND = '0.05';       // m: a limb further behind the cloth's surface than its radius + this is not pressing it
const PUSH = `
#include <skinning_vertex>
{
  vec4 hsinW = modelMatrix * vec4( transformed, 1.0 );
  bool hsinMoved = false;
  vec3 hsinBN = hsinButtW.x + hsinButtW.y + hsinButtL.x + hsinButtL.y > 0.0 ? normalize( hsinHipsRot * hsinButtN ) : vec3( 0.0 );
  if ( hsinButtW.x + hsinButtW.y > 0.0 ) {   // first the sway, then the limbs push (a hand on the bottom still presses it)
    hsinW.xyz += hsinButtW.x * ( hsinButt[ 0 ] + hsinButtUp * hsinButtShape[ 0 ].x + hsinBN * hsinButtShape[ 0 ].y )
               + hsinButtW.y * ( hsinButt[ 1 ] + hsinButtUp * hsinButtShape[ 1 ].x + hsinBN * hsinButtShape[ 1 ].y );
    hsinMoved = true;
  }
  if ( hsinButtL.x + hsinButtL.y > 0.0 ) {   // loose cloth over the bottom: pushed out along its own outward direction
    vec3 n = hsinBN;                                 // when the flesh (+ the tight layer) moves out into it
    vec3 f0 = hsinButt[ 0 ] + hsinButtUp * hsinButtShape[ 0 ].x + n * hsinButtShape[ 0 ].y;
    vec3 f1 = hsinButt[ 1 ] + hsinButtUp * hsinButtShape[ 1 ].x + n * hsinButtShape[ 1 ].y;
    float e = dot( hsinButtL.x * f0 + hsinButtL.y * f1, n );
    float push = e + ${BUTT_PAD_GLSL} - hsinButtL.z - dot( hsinButtW.x * f0 + hsinButtW.y * f1, n );
    if ( e > 0.0 && push > 0.0 ) { hsinW.xyz += n * min( push, e ); hsinMoved = true; }
  }
  for ( int i = 0; i < ${N}; i ++ ) {
    float r = hsinA[ i ].w;
    float code = hsinB[ i ].w;
    if ( r <= 0.0 || abs( hsinOwner - code ) < 0.5 || hsinOwner > 5.5 ) continue;   // 6 = head / neck: never pushed
    if ( code > 2.5 ) {                       // a leg: clothes only, minus their rest depth
      if ( hsinCloth < 0.5 ) continue;
      // sleeves (cloth carried by an arm) are not pushed by the legs / bottom: the solver keeps the arm itself out of
      // them, and the push moved a hanging sleeve's inner side through its own forearm (test1 at rest, the author 2026-11-11)
      if ( hsinOwner > 0.5 && hsinOwner < 2.5 ) continue;
      int s = int( hsinSlot[ i ] + 0.5 );
      float rest = s == 0 ? hsinRest.x : s == 1 ? hsinRest.y : s == 2 ? hsinRest.z : s == 3 ? hsinRest.w : hsinRestB;
      r -= rest;
      if ( r <= 0.0 ) continue;
    }
    vec3 a = hsinA[ i ].xyz, ab = hsinB[ i ].xyz - a;
    if ( code < 2.5 && hsinCloth > 2.5 && hsinOwner < 0.5 ) {   // front-facing trunk cloth only (3)
      // Loose cloth on the trunk with an arm between it and the body: the cloth goes BEHIND the arm, pressed onto the
      // body, as a shirt does when a forearm crosses the chest (2026-11-10, the author: the forearm went into the T-shirt).
      // Along the vertex's own outward normal: where that column passes through the capsule, the vertex moves back to
      // just behind the capsule's far side. An arm wholly in front of the cloth leaves it alone. Only vertices whose
      // normal faces OUT from the trunk (hsinCloth 2) and a limb at most CLOTH_BEHIND behind the surface: linings and
      // cuff insides face inward, and with a window of 30 cm a hanging hand pulled them into spikes toward it (the
      // default model's shorts looked torn, the author 2026-11-10).
      vec3 n = normalize( ( modelMatrix * vec4( objectNormal, 0.0 ) ).xyz );
      vec3 r0 = a - hsinW.xyz;
      float a11 = max( dot( ab, ab ), 1e-8 ), a12 = -dot( ab, n ), det = a11 - a12 * a12;
      float u = det > 1e-6 * a11 ? ( -dot( r0, ab ) + a12 * dot( r0, n ) ) / det : -dot( r0, ab ) / a11;
      vec3 cc = a + clamp( u, 0.0, 1.0 ) * ab, vc = hsinW.xyz - cc;
      vec3 w = vc - n * dot( vc, n );
      float wl = length( w ), back = dot( vc, n );
      // only a limb in front of the trunk (within its half width sideways): beside it (a hand holding the other upper
      // arm) the rule folded the shirt's side behind the forearm and showed its inside (2026-11-10)
      bool inFront = abs( dot( cc - hsinTrunk.xyz, hsinTrunkX ) ) < hsinTrunk.w && abs( dot( hsinB[ i ].xyz - hsinTrunk.xyz, hsinTrunkX ) ) < hsinTrunk.w;   // ... and the limb does not reach past the trunk's side (a hand holding the other arm)
      if ( inFront && wl < r + ${CLOTH_DENT} && back > 0.0 && back < r + ${CLOTH_BEHIND} ) {   // back > 0: only a limb whose axis is BEHIND the cloth surface (inside the shirt) moves it; one merely touching from the front dragged a loose shirt's side 3 cm back and tore it (2026-11-10)
        // inside the column: to the capsule's far side; beside it: a dent fading out over CLOTH_DENT (a loose shirt's
        // big triangles made dark pits when only the column moved)
        float s2 = wl < r ? back + sqrt( r * r - wl * wl ) : back * ( 1.0 - ( wl - r ) / ${CLOTH_DENT} );
        if ( s2 > 0.0 ) { hsinW.xyz -= n * ( min( s2, ${CLOTH_MAX} ) + 0.004 ); hsinMoved = true; continue; }   // capped: deeper, a coarse shirt's unmoved neighbours turned the moved triangles inside out (a black patch; 2026-11-10)
      }
    }
    float t = clamp( dot( hsinW.xyz - a, ab ) / max( dot( ab, ab ), 1e-8 ), 0.0, 1.0 );
    vec3 c = a + t * ab, d = hsinW.xyz - c;
    float l = length( d );
    if ( l >= r || l <= 1e-5 ) continue;
    vec3 pn = hsinPalm[ i ];
    if ( hsinCloth > 0.5 && dot( pn, pn ) > 0.5 ) {
      // cloth under a hand goes to the palm side (the hand presses it onto what it rests on); pushed round the hand
      // it came up over the fingers (2026-10-29, test1: hands resting on the skirt looked sunk in it)
      vec3 side = d - pn * dot( d, pn );
      hsinW.xyz = c + side + pn * sqrt( max( r * r - dot( side, side ), 0.0 ) );
    } else {
      hsinW.xyz += d * ( r / l - 1.0 );
    }
    hsinMoved = true;
  }
  if ( hsinMoved ) transformed = ( inverse( modelMatrix ) * hsinW ).xyz;
}
`;

function limbOf(name) {                   // limb code of a humanoid bone (0: not a pushing limb's chain)
  if (!name) return 0;
  // the head and neck (face, hair) are never pushed (2026-11-03: fists under the chin dented the face; the solver keeps
  // the hands on the face's surface, a pressed-in face looked broken)
  if (/^(head|neck|jaw|leftEye|rightEye)$/.test(name)) return 6;
  for (const s of ['left', 'right']) {
    if (!name.startsWith(s)) continue;
    const rest = name.slice(s.length);
    if (/^(UpperArm|LowerArm|Hand|Thumb|Index|Middle|Ring|Little)/.test(rest)) return CODE[`${s}arm`];
    if (/^(UpperLeg|LowerLeg|Foot|Toes)/.test(rest)) return CODE[`${s}leg`];
  }
  return 0;
}

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _v = new THREE.Vector3();

function capsuleWorld(l) {               // world end points of a measured limb capsule (hand stretched to the tip)
  l.bone.updateWorldMatrix(true, false);
  _a.copy(l.a).applyMatrix4(l.bone.matrixWorld);
  _b.copy(l.b).applyMatrix4(l.bone.matrixWorld);
  if (l.name.endsWith('Hand')) _b.sub(_a).multiplyScalar(HAND_TIP).add(_a);
  return [_a, _b];
}

// The bottom (2026-11-03, test1 knees_hugged: the back of a skirt skinned to both thighs is pulled in at its middle
// when the thighs rise, and the bottom, behind the hip joints and outside the thigh capsules, came through it). A
// capsule across the hips, carried by the hips bone, from the skin at rest: axis through the hip joints, from one
// to the other, radius = 90th percentile distance (in the side view) of the skin behind the hip joints between them.
// It pushes clothes only, minus their rest depth, like the leg capsules.
function bottomCapsule(vrm, skin, humanOf) {
  const H = vrm.humanoid, hips = H.getRawBoneNode('hips');
  if (!skin?.geometry.index || !hips) return null;
  const L = H.getRawBoneNode('leftUpperLeg').getWorldPosition(new THREE.Vector3());
  const R = H.getRawBoneNode('rightUpperLeg').getWorldPosition(new THREE.Vector3());
  const knee = H.getRawBoneNode('leftLowerLeg').getWorldPosition(new THREE.Vector3()), thigh = L.y - knee.y;
  const c = L.clone().add(R).multiplyScalar(0.5), hx = Math.abs(L.x - R.x) / 2;
  const g = skin.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight, v = new THREE.Vector3(), d = [];
  for (const i of new Set(g.index.array)) {
    let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
    if (!/^(hips|leftUpperLeg|rightUpperLeg)$/.test(humanOf(skin.skeleton.bones[sk.getComponent(i, best)]) ?? '')) continue;
    skin.getVertexPosition(i, v); v.applyMatrix4(skin.matrixWorld);
    if (Math.abs(v.x - c.x) > hx || v.z > c.z || v.y > c.y + 0.3 * thigh || v.y < c.y - 0.5 * thigh) continue;
    d.push(Math.hypot(v.y - c.y, v.z - c.z));
  }
  if (d.length < 20) return null;
  d.sort((x, y) => x - y);
  const inv = hips.matrixWorld.clone().invert();
  const skinR = d[Math.floor(0.9 * (d.length - 1))];
  return { bone: hips, name: 'bottom', code: CODE.hips, slot: 4, radius: skinR, skinR,
    a: new THREE.Vector3(c.x - hx, c.y, c.z).applyMatrix4(inv), b: new THREE.Vector3(c.x + hx, c.y, c.z).applyMatrix4(inv) };
}

// Buttock sway weights (2026-11-05, the author; no model has bones there, so the sway is a displacement in this shader).
// By shape, at rest: vertices of the trunk / thighs (strongest bone, through spring joints), behind the hip joints; per side an ellipse
// in the frontal plane round the buttock's centre (BUTT_CENTRE thighs below the hip joint, i.e. above for < 0, BUTT_RX half the hip-joint
// distance wide, BUTT_RY thigh lengths tall), faded to 0 toward its rim, toward the hip joints' plane (BUTT_DEPTH m
// behind it = full) and upward to the waist (BUTT_TOP thighs above the hip joints: full .. none; the waistband,
// where layers of clothes overlap, stays put). Skin and clothes in the region get the same weight from their rest
// position. Only the flesh and what hugs it follow (2026-11-05, the author: 'the bottom should sway, not the clothes'):
// times a tightness factor from the rest distance outside the skin's bottom capsule (BUTT_TIGHT: none at 3 cm, full
// at 1.5 cm and closer; VRoid deletes the skin under clothes, so the measured capsule stands in for it). Loose clothes
// stay; when the flesh moving out (+ BUTT_PAD for the tight layer) reaches one, it is pushed out by that much along
// its own outward direction off the bottom (per vertex in the shader: hsinButtL = the flesh's weights there + the
// rest gap, hsinButtN = that direction; only pushing straight back let the default model's shorts come through its
// T-shirt on the upper curve of the bottom when stepping in place, 2026-11-05). Moving the bottom capsule instead
// striped AvatarSample_D's layered skirt at its sides (2026-11-05). (2026-11-05, three tries: leaving a whole mesh with a
// spring vertex still let the default model's shorts come through its T-shirt at 1 cm back; fading out near spring
// vertices left the 0.x sample's bottom still under its long T-shirt's hem and only the shirt above it moved, which
// read as the waist swaying; moving only a shirt's static part tore it from its hem.) Returns the two centres.
// (2026-11-05, the author: 'it is the thighs that sway'): the first values (centre 0.12 thighs BELOW the hip joints, cut at
// their height) covered the backs of the thighs. Measured back profiles (default, test2, AvatarSample_D): the buttock
// bulges most from the hip joints' height to 0.2 thighs above them; below -0.05 it is the thigh. Centred 0.08 above
// with the fade from 0.25 the 0.x sample's lower back moved (the author: 'the waist sways'): centre at the joints' height.
// The field is kept smooth (wide fades): a coarse shirt's big triangles interpolate it linearly while dense shorts
// under it follow its curve, so a steep field let the shorts come through at 2 cm (default model, 'strong').
const BUTT_PAD = 0.03, BUTT_TIGHT = [0.03, 0.015];
const BUTT_CENTRE = 0.0, BUTT_RX = 1.3, BUTT_RY = 0.3, BUTT_DEPTH = 0.08, BUTT_TOP = [0.0, 0.3], BUTT_EDGE = [1.0, 0.0];
function buttWeights(vrm, o, humanOf, out, bottom, loose, viaSpring, normal, rec) {
  const H = vrm.humanoid, hips = H.getRawBoneNode('hips');
  const L = H.getRawBoneNode('leftUpperLeg').getWorldPosition(new THREE.Vector3());
  const R = H.getRawBoneNode('rightUpperLeg').getWorldPosition(new THREE.Vector3());
  const knee = H.getRawBoneNode('leftLowerLeg').getWorldPosition(new THREE.Vector3()), thigh = L.y - knee.y;
  const hx = Math.abs(L.x - R.x) / 2, cz = (L.z + R.z) / 2;
  const cen = [L, R].map((p) => new THREE.Vector3(p.x, p.y - BUTT_CENTRE * thigh, p.z));
  const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight, n = g.attributes.position.count;
  const sm = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
  const used = g.index ? new Set(g.index.array) : null, hipY = Math.min(L.y, R.y);
  const region = []; let swings = false;
  const hipsInv = hips.matrixWorld.clone().invert();
  for (let i = 0; i < n; i++) {
    if (used && !used.has(i)) continue;
    let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
    const bone = o.skeleton.bones[sk.getComponent(i, best)];
    // the body's trunk and thighs (the region itself is set by position below; this only leaves out arms / hands
    // hanging beside the bottom); spine / chest too: the default model's T-shirt hangs from the spine over its shorts
    if (!/^(hips|spine|chest|upperChest|leftUpperLeg|rightUpperLeg)$/.test(humanOf(bone) ?? '')) continue;
    o.getVertexPosition(i, _v); _v.applyMatrix4(o.matrixWorld);
    const behind = cz - _v.z;                  // world at load: every model faces +z (rotateVRM0 turns 0.x models)
    if (behind <= 0) continue;
    const top = sm(hipY + BUTT_TOP[1] * thigh, hipY + BUTT_TOP[0] * thigh, _v.y);
    if (!top) continue;
    const depth = sm(0, BUTT_DEPTH, behind);
    let w = [0, 1].map((s) => { const c = cen[s]; return top * depth * sm(BUTT_EDGE[0], BUTT_EDGE[1], Math.hypot((_v.x - c.x) / (BUTT_RX * hx), (_v.y - c.y) / (BUTT_RY * thigh))); });
    let tight = 1, gapOut = 0;
    if (bottom) {
      const ab = _b.copy(bottom.b).sub(bottom.a), t = Math.min(1, Math.max(0, _a.copy(_v).sub(bottom.a).dot(ab) / ab.lengthSq()));
      gapOut = Math.max(0, _v.distanceTo(_a.copy(bottom.a).addScaledVector(ab, t)) - bottom.skinR);
      tight = sm(BUTT_TIGHT[0], BUTT_TIGHT[1], gapOut);
    }
    out[2 * i] = w[0] * tight; out[2 * i + 1] = w[1] * tight;
    if (bottom && tight < 1) { loose[3 * i] = w[0]; loose[3 * i + 1] = w[1]; loose[3 * i + 2] = gapOut; }
    region.push([i, w[0], w[1], gapOut, _v.clone()]);
    if (bottom) {                                // outward off the bottom capsule's axis, in the hips' frame
      const ab = _b.copy(bottom.b).sub(bottom.a), t = Math.min(1, Math.max(0, _a.copy(_v).sub(bottom.a).dot(ab) / ab.lengthSq()));
      const nrm = _a.copy(_v).sub(_b.copy(bottom.a).addScaledVector(bottom.b.clone().sub(bottom.a), t)).normalize().transformDirection(hipsInv);
      normal[3 * i] = nrm.x; normal[3 * i + 1] = nrm.y; normal[3 * i + 2] = nrm.z;
    }
    if (viaSpring(bone)) swings = true;
  }
  // a garment with a free hem here (spring-driven vertices in the region: a T-shirt, a skirt) is loose as a whole,
  // however close it lies (the 0.x sample's long T-shirt rests on the bottom and swayed with it, 2026-11-05, the author:
  // 'clothes should not sway like that'): it only gets pushed
  if (swings && bottom) for (const [i, w0, w1, gap] of region) { out[2 * i] = 0; out[2 * i + 1] = 0; loose[3 * i] = w0; loose[3 * i + 1] = w1; loose[3 * i + 2] = gap; }
  for (const [i, , , , p] of region) {          // for coverBottom: who follows, who covers
    if (out[2 * i] || out[2 * i + 1]) rec.tight.push({ out, i, p });
    if (loose[3 * i] || loose[3 * i + 1]) rec.loose.push({ loose, i, p });
  }
  const inv = hips.matrixWorld.clone().invert();
  return cen.map((c) => c.clone().applyMatrix4(inv));
}

// Flesh under loose clothes sways less (2026-11-05): a coarse T-shirt pushed out over moving shorts still let them
// through between its big triangles when stepping in place. A following vertex fades with its distance to the nearest
// loose cloth vertex round the bottom (BUTT_COVER: none at 1.5 cm, full at 5 cm); a loose vertex then takes the factor
// of the nearest following vertex for its push (the flesh under it moves that much). Only what shows sways fully.
const BUTT_COVER = [0.015, 0.05];
function coverBottom(rec) {
  const sm = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
  if (!rec.loose.length || !rec.tight.length) return;
  const L = rec.loose.map((r) => r.p), T = rec.tight.map((r) => r.p);
  const near = (p, list) => { let d2 = Infinity; for (const q of list) { const d = p.distanceToSquared(q); if (d < d2) d2 = d; } return Math.sqrt(d2); };
  const f = rec.tight.map((r) => sm(BUTT_COVER[0], BUTT_COVER[1], near(r.p, L)));
  rec.loose.forEach((r) => {
    let best = Infinity, k = 0;
    T.forEach((q, j) => { const d = r.p.distanceToSquared(q); if (d < best) { best = d; k = j; } });
    r.loose[3 * r.i] *= f[k]; r.loose[3 * r.i + 1] *= f[k];
  });
  rec.tight.forEach((r, j) => { r.out[2 * r.i] *= f[j]; r.out[2 * r.i + 1] *= f[j]; });
}

export function addLimbPush(vrm) {
  const H = vrm.humanoid;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const humanOf = (node) => { let n = node; while (n && !rawToName.has(n)) n = n.parent; return n ? rawToName.get(n) : null; };
  const limbs = (vrm.scene.userData.hsinLimbs ?? []).map((l) => ({ ...l, name: rawToName.get(l.bone) }))
    .filter((l) => /^(left|right)(LowerArm|Hand|UpperLeg|LowerLeg)$/.test(l.name ?? '')).slice(0, N);
  for (const l of limbs) {
    const side = l.name.startsWith('left') ? 'left' : 'right', kind = LIMB_BONES[l.name.slice(side.length)];
    l.code = CODE[side + kind];
    l.slot = kind === 'leg' ? (side === 'left' ? 0 : 2) + (l.name.endsWith('LowerLeg') ? 1 : 0) : -1;
  }
  vrm.scene.userData.hsinPushLimbs = limbs;
  const uniforms = {
    hsinA: { value: Array.from({ length: N }, () => new THREE.Vector4()) },
    hsinB: { value: Array.from({ length: N }, () => new THREE.Vector4()) },
    hsinSlot: { value: Array.from({ length: N }, () => -1) },
    hsinPalm: { value: Array.from({ length: N }, () => new THREE.Vector3()) },
    hsinButt: { value: [new THREE.Vector3(), new THREE.Vector3()] },
    hsinButtBack: { value: new THREE.Vector3(0, 0, -1) },
    hsinHipsRot: { value: new THREE.Matrix3() },
    hsinTrunk: { value: new THREE.Vector4(0, 0, 0, 0.12) },
    hsinTrunkX: { value: new THREE.Vector3(1, 0, 0) },
    hsinButtShape: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    hsinButtUp: { value: new THREE.Vector3(0, 1, 0) },
  };
  const springs = new Set([...(vrm.springBoneManager?.joints ?? [])].map((j) => j.bone)), viaMemo = new Map();
  const viaSpring = (b) => {                    // the bone is a spring joint or hangs from one (below the humanoid bone)
    if (viaMemo.has(b)) return viaMemo.get(b);
    let n = b, r = false; while (n && !rawToName.has(n)) { if (springs.has(n)) { r = true; break; } n = n.parent; }
    viaMemo.set(b, r); return r;
  };
  let buttCentres = null;
  const buttRec = { tight: [], loose: [] };
  vrm.humanoid.resetNormalizedPose?.(); vrm.update?.(0); vrm.scene.updateMatrixWorld(true);
  const skin = skinMesh(vrm);
  const trunk = new THREE.Vector3(); vrm.humanoid.getNormalizedBoneNode('hips').getWorldPosition(trunk);
  const uaN = vrm.humanoid.getNormalizedBoneNode('leftUpperArm'), halfW = uaN ? Math.abs(uaN.getWorldPosition(new THREE.Vector3()).x - trunk.x) : 0.13;   // shoulder half width (rest)
  const bottom = bottomCapsule(vrm, skin, humanOf);
  if (bottom && limbs.length < N) limbs.push(bottom);
  const bottomW = bottom ? (() => { const [a, b] = capsuleWorld(bottom); return { a: a.clone(), b: b.clone(), skinR: bottom.skinR }; })() : null;
  const legCaps = limbs.filter((l) => l.slot >= 0).map((l) => { const [a, b] = capsuleWorld(l); return { a: a.clone(), b: b.clone(), r: l.radius, slot: l.slot, code: l.code }; });
  const patched = new Set();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight, n = g.attributes.position.count;
    const cloth = o !== skin;
    // attributes per geometry; VRoid primitives share one vertex buffer but each has its own geometry object
    const own = new Float32Array(n), rest = new Float32Array(4 * n), restB = new Float32Array(n), clo = new Float32Array(n);
    const nrmAttr = g.attributes.normal, nrmMat = new THREE.Matrix3().getNormalMatrix(o.matrixWorld);
    for (let i = 0; i < n; i++) {
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      own[i] = limbOf(humanOf(o.skeleton.bones[sk.getComponent(i, best)]));
      if (!cloth) continue;
      o.getVertexPosition(i, _v); _v.applyMatrix4(o.matrixWorld);
      // 2 = the rest normal faces out from the trunk's vertical axis (no normal attribute: treated as outward)
      clo[i] = 2;
      if (nrmAttr) {
        _a.fromBufferAttribute(nrmAttr, i).applyMatrix3(nrmMat);
        if (_a.x * (_v.x - trunk.x) + _a.z * (_v.z - trunk.z) < 0) clo[i] = 1;
        else if (_a.z > Math.hypot(_a.x, _a.z) * Math.cos(CLOTH_FRONT_DEG * Math.PI / 180) && Math.abs(_v.x - trunk.x) < CLOTH_FRONT_W * halfW) clo[i] = 3;   // faces forward, in the middle of the trunk's width
      }
      for (const c of legCaps) {
        if (c.code === own[i]) continue;
        const ab = c.b.clone().sub(c.a), t = Math.min(1, Math.max(0, _v.clone().sub(c.a).dot(ab) / ab.lengthSq()));
        const d = _v.distanceTo(c.a.clone().add(ab.multiplyScalar(t)));
        if (c.slot < 4) rest[4 * i + c.slot] = Math.max(0, c.r - d); else restB[i] = Math.max(0, c.r - d);
      }
    }
    g.setAttribute('hsinOwner', new THREE.BufferAttribute(own, 1));
    g.setAttribute('hsinCloth', new THREE.BufferAttribute(clo, 1));
    g.setAttribute('hsinRest', new THREE.BufferAttribute(rest, 4));
    g.setAttribute('hsinRestB', new THREE.BufferAttribute(restB, 1));
    const bw = new Float32Array(2 * n), bl = new Float32Array(3 * n), bn = new Float32Array(3 * n);
    buttCentres = buttWeights(vrm, o, humanOf, bw, bottomW, bl, viaSpring, bn, buttRec);
    g.setAttribute('hsinButtN', new THREE.BufferAttribute(bn, 3));
    g.setAttribute('hsinButtW', new THREE.BufferAttribute(bw, 2));
    g.setAttribute('hsinButtL', new THREE.BufferAttribute(bl, 3));
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (!m || patched.has(m)) continue;
      patched.add(m);
      const prev = m.onBeforeCompile;
      m.onBeforeCompile = (shader, renderer) => {
        if (prev) prev.call(m, shader, renderer);
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = DECL + shader.vertexShader.replace('#include <skinning_vertex>', PUSH);
      };
      const key = m.customProgramCacheKey?.bind(m);
      m.customProgramCacheKey = () => (key ? key() : '') + '|hsinLimbPush';
      m.needsUpdate = true;
    }
  });
  coverBottom(buttRec);
  vrm.scene.userData.hsinPush = uniforms;
  vrm.scene.userData.hsinButt = buttCentres ? { bone: vrm.humanoid.getRawBoneNode('hips'), centres: buttCentres } : null;
  return patched.size;
}

// The way a palm faces in the world: the normalized hand bone's -y (VRM's T-pose has the palms down), through the
// normalized node so 0.x models (turned) agree.
const _q = new THREE.Quaternion();
function palmDir(vrm, l) {
  const n = vrm.humanoid.getNormalizedBoneNode(l.name);
  n.getWorldQuaternion(_q);
  return _v.set(0, -1, 0).applyQuaternion(_q).normalize();
}

export function updateLimbPush(vrm) {
  const u = vrm.scene.userData.hsinPush, limbs = vrm.scene.userData.hsinPushLimbs;
  if (!u || !limbs) return;
  let k = 0;
  for (const l of limbs) {
    const [a, b] = capsuleWorld(l);
    u.hsinA.value[k].set(a.x, a.y, a.z, l.radius);
    u.hsinB.value[k].set(b.x, b.y, b.z, l.code);
    u.hsinSlot.value[k] = l.slot;
    if (l.name.endsWith('Hand')) u.hsinPalm.value[k].copy(palmDir(vrm, l)); else u.hsinPalm.value[k].set(0, 0, 0);
    k++;
  }
  for (; k < N; k++) u.hsinA.value[k].w = 0;
  const H = vrm.humanoid, uc = H.getNormalizedBoneNode('upperChest') || H.getNormalizedBoneNode('chest'), ua = H.getNormalizedBoneNode('leftUpperArm');
  if (uc) {
    uc.getWorldPosition(_a); u.hsinTrunk.value.set(_a.x, _a.y, _a.z, u.hsinTrunk.value.w);
    _b.set(1, 0, 0).transformDirection(uc.matrixWorld); u.hsinTrunkX.value.copy(_b);
    if (ua) { ua.getWorldPosition(_v); u.hsinTrunk.value.w = Math.max(0.06, Math.abs(_v.clone().sub(_a).dot(_b)) - 0.02); }
  }
}
