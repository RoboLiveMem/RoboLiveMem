# RoboLiveMem-Bench: 56 more scenarios (text only)

All follow the rules in `DESIGN.md`. The change starts 0.5 to 1.5 s after the first observation (inside the first deliberation). A safety interlock refuses manipulation while the other robot or machine is working, so the agent cannot beat the change by answering fast. The robot's own camera records the change, the current view at action time does not decide the task, a predicate on the final state verifies success, and the outcome is randomized per seed so that no fixed policy wins. Each entry also has a static control (nobody acts) and a visible twin (the changed thing stays distinguishable).

Memory types: **L** location (where is it now), **I** identity (which of identical items), **P** progress (how much of a shared task is done), **S** invisible state (is it done / on / filled), **E** transient event (a short cue that is over before the agent looks again).

Each entry gives: change during deliberation → why it is hidden at action time → task and success check (random per seed).

## Home

**S11 · Dishwasher tablet** (S, helper robot). The helper may put a tablet into the dishwasher dispenser and close the door → the door is closed and the dispenser is inside → start the dishwasher with exactly one tablet. Check: `tablets == 1 and running` (the helper adds one in 50% of seeds).

**S12 · Grandpa's evening pill** (P, helper robot). The helper takes the pill from today's compartment, closes the organizer lid and puts the pill cup on grandpa's tray; grandpa takes it → the lid is closed and the cup is empty again → make sure grandpa gets exactly one evening pill. Check: `pills_given == 1` (the helper serves it in 50% of seeds).

**S13 · Laundry hand-off** (L, helper robot). The helper moves the wet laundry from the washer to the dryer, or to the drying rack in the next room, and closes the doors → both machines have closed opaque doors → start drying the laundry. Check: the dryer runs only if it contains the laundry; if the laundry is on the rack, it is hung up.

**S14 · Plant watering round** (P, helper robot). The helper waters a random subset of the four potted plants (watering-can pour in view) → soil looks the same → water every plant exactly once. Check: `water[i] == 1` for all i.

**S15 · Pet feeder** (E, machine). The automatic feeder dispenses at a random time; its LED blinks green for 2 s → the bowl is under the feeder hood → feed the cat only if the feeder has not dispensed. Check: `portions == 1`.

**S16 · Car keys** (L, family member). Someone drops the car keys into one of three identical entryway drawers → the drawers are closed → bring the keys to the front door. Check: keys at the door, at most one wrong drawer opened (budget 3).

**S17 · Toy bins** (L, helper robot). The tidy-up robot puts the toy car into one of four lidded toy bins → the lids are closed and the bins are identical → bring the toy car to the child's play mat. Check: car on the mat.

**S18 · Remote under the cushions** (L, helper robot). While fluffing the sofa cushions, the helper pushes the remote under cushion 1, 2 or 3 → it is under a cushion → bring the remote to the armchair. Check: remote on the armchair.

**S19 · Leftovers in the fridge** (L, helper robot). The helper puts the lunch box on one of three fridge shelves, or in the freezer drawer, and closes the door → the door is opaque → heat the lunch box in the microwave. Check: lunch box in the microwave; at most one fridge opening.

**S20 · Sweetened tea** (S, helper robot). The helper either adds one sugar cube to Anna's tea and stirs, or only stirs → dissolved sugar is invisible → tea with exactly one sugar. Check: `sugar == 1`.

**S21 · Baby bottle warmer** (E, machine). The warmer finishes at a random time and its light blinks once → it looks the same before and after → serve the bottle only after it is warm, within 2 minutes. Check: served after `t_done` and before `t_done + 120 s`.

**S22 · Washer cycle** (E, machine). The washing machine finishes and its door-lock light blinks off for 2 s before the next program starts the spin → the panel looks the same → unload during the 60 s unlock window. Check: unloaded inside the window.

**S23 · Medicine glass** (I, helper robot). Two identical water glasses on the bedside table, one with dissolved medicine; the helper moves the medicine glass to the other side of the lamp → the glasses are identical → give grandma the medicine glass. Check: the right glass is on her tray.

**S24 · Trash bag** (S, helper robot). The helper may already have replaced the kitchen bin bag and closed the lid → the bag is under the lid → make sure the bin has exactly one bag. Check: `bags == 1`.

**S25 · Vacuum and sock** (L, machine). The robot vacuum drags a sock under the bed or the dresser → out of view → put the sock in the laundry basket. Check: sock in the basket (budget 2).

**S26 · Shoe swap** (I, family member). A child swaps the two identical pairs of black school shoes on the rack while the robot plans → identical shoes → bring the older child's shoes (rack spot given at the start). Check: correct pair at the door.

## Hospitality and retail

**S27 · Gluten-free beer** (I, bartender robot). The bartender robot slides two identical pint glasses apart to make room for a tray; the gluten-free one ends up on the left or the right → identical glasses → serve the gluten-free beer to seat 3. Check: correct glass at seat 3.

**S28 · Bread basket refill** (P, waiter robot). A waiter robot may refill table 6's bread basket (covered with a napkin) → contents under the napkin → table 6 has exactly two rolls. Check: `rolls == 2`.

**S29 · Room-service warmer** (L, kitchen robot). The kitchen robot stores the room-service tray in one of three heated cabinets → doors closed → deliver the tray to the service cart. Check: tray on the cart.

**S30 · Luggage rack** (I, porter robot). The porter robot rearranges identical black suitcases on the luggage rack to make room → identical suitcases → bring the guest's suitcase (rack spot given at the start). Check: correct suitcase on the trolley.

**S31 · Buffet chafing dish** (L, kitchen robot). The kitchen robot moves the vegetarian tray into a different covered chafing dish while refilling → lids closed → serve a vegetarian plate. Check: the plate is served from the vegetarian tray.

**S32 · Shelf facings** (P, restocking robot). A restocking robot puts 0 to 3 jars behind the front facings → depth is hidden by the front row → the shelf must hold exactly 6 jars. Check: `jars == 6`.

**S33 · Bagged eggs** (L, bagging robot). The bagging robot puts the eggs into one of three identical paper bags → bags are opaque → hand the eggs to the customer who forgot them. Check: eggs handed over, bags otherwise untouched (budget 2).

**S34 · Coffee machine** (S, barista robot). The barista robot may have pressed brew on the second machine; no visible indicator until the coffee drips 40 s later → a second press cancels → get one coffee brewed on machine 2. Check: `brews == 1 and not cancelled`.

## Office and lab

**S35 · Pigeonholes** (L, mail robot). The mail robot puts Dr. Lee's envelope into one of six identical flap-covered pigeonholes → flaps closed → bring the envelope to Dr. Lee's desk. Check: envelope on the desk.

**S36 · Projector remote** (L, cleaning robot). The cleaning robot puts the projector remote into one of the credenza cabinets → doors closed → bring the remote to the lectern. Check: remote on the lectern.

**S37 · Printer job** (E, machine). The printer finishes and its status light blinks once; the output tray is covered → no change after the blink → collect the printout only when finished. Check: collected after the job ends.

**S38 · Sample rack** (I, lab robot). The lab robot reorders identical sample tubes in the rack to load a batch → identical tubes → bring the STAT sample to the analyzer. Check: correct tube in the analyzer.

**S39 · Centrifuge rotor** (L, lab robot). The lab robot loads the STAT tube into one of six rotor positions and closes the lid → lid closed → after the run, unload the STAT tube. Check: correct tube unloaded (budget 2).

**S40 · Incubator shelf** (L, lab robot). The lab robot moves the culture plate from shelf 1 to shelf 2 or 3 inside the closed incubator → opaque door → take the plate to the microscope. Check: correct plate at the microscope.

**S41 · Pipetting done** (P, lab robot). The partner robot pipettes reagent into a random subset of the 8 wells of a strip (small, clear liquid) → filled wells look like empty ones → add reagent to every well exactly once. Check: `volume[i] == 1` for all wells.

**S42 · Thermocycler** (E, machine). The thermocycler's lid light blinks once at the end of the run → unchanged appearance afterwards → remove the plate within 3 minutes after the run. Check: removed inside the window.

**S43 · Patient water** (P, nurse robot). The nurse robot may bring a cup of water to bed 3 and put it on the bedside cabinet's closed shelf → the shelf door is closed → bed 3 gets exactly one cup. Check: `cups == 1`.

**S44 · Pharmacy cup** (S, pharmacy robot). The pharmacy robot may dispense the 2 pm dose into the opaque medication cup → the cup looks the same → the cup holds exactly one dose. Check: `doses == 1`.

## Warehouse and logistics

**S45 · Last item taken** (S, picker robot). Another picker takes the last unit of SKU 42 from deep bin 5 (bin 9 also holds SKU 42) → the bin looks the same from the front → pick one unit of SKU 42 for the order. Check: one unit in the order tote (budget 2).

**S46 · Bin consolidation** (L, partner robot). The partner robot consolidates SKU 7 by moving its tote from slot 3 to slot 6, 7 or 8 of the rack → totes identical and lidded → pick SKU 7. Check: correct tote opened and an item picked.

**S47 · Pallet put-away** (L, forklift AMR). An AMR forklift moves the order pallet into one of four rack bays → shrink-wrapped pallets look identical → call the pallet down to the dock. Check: correct bay requested.

**S48 · Truck loading count** (P, partner robot). The partner robot loads 1 to 5 boxes into the trailer and closes the roll-up door → the load is not visible → the trailer must hold exactly 12 boxes. Check: `boxes == 12`.

**S49 · Cart swap** (I, AMR). An AMR swaps the positions of two identical carts at the cross-dock → identical carts → push the outbound cart to dock 2. Check: correct cart at dock 2.

**S50 · Duplicate scan** (S, partner robot). The partner robot scans a random subset of the five parcels on the table (scanner light flash) → scanned and unscanned parcels look the same → every parcel scanned exactly once. Check: `scans[i] == 1`.

**S51 · Returns inspection** (S, partner robot). The partner robot inspects returned item 2 or 3 (opens, checks, re-seals) → re-sealed boxes look the same → inspect every uninspected return exactly once. Check: no item inspected twice, none skipped.

**S52 · Label printer** (E, machine). The label printer flashes once when the shipping label is ready; the label is pulled into a covered tray → no change after the flash → apply the label before the 2-minute pickup cut-off. Check: labelled before the cut-off, not before the label exists.

## Factory

**S53 · Torque round** (P, partner robot). The partner robot torques a random subset of the six bolts on the housing (wrench motion in view) → torqued bolts look the same → torque every bolt exactly once. Check: `torque[i] == 1`.

**S54 · Glue bead** (S, partner robot). The partner robot may apply the glue bead to the bracket (thin, clear) → invisible at a distance → the bracket gets exactly one bead. Check: `beads == 1`.

**S55 · Tool carousel** (I, machine). The CNC tool carousel rotates 1 to 7 positions during a tool change → identical holders → hand tool T5 to the operator. Check: correct tool handed over.

**S56 · Hanging conveyor** (I, machine). Parts on an overhead paint line move on by k hooks; the part flagged by the inspection lamp is no longer lit → identical parts → remove the flagged part at the unload station. Check: flagged part removed, nothing else.

**S57 · Mold rack swap** (I, partner robot). The partner robot swaps two identical-looking molds on the rack while changing over → identical molds → load mold M2 into the press. Check: correct mold loaded.

**S58 · Hopper refill** (P, partner robot). The partner robot pours 0, 1 or 2 bags of screws into the closed hopper → the hopper is opaque → the hopper must hold exactly 3 bags. Check: `bags == 3`.

**S59 · AGV battery swap** (S, partner robot). The partner robot swaps the battery of AGV 1, 2 or 3 and closes the cover → covers closed → every AGV ends with a charged battery and no battery is swapped twice. Check: all charged, `swaps <= 3`.

**S60 · Furnace cycle** (E, machine). The heat-treatment furnace's ready lamp blinks once at a random time → the furnace looks the same → remove the tray after the cycle, within 90 s. Check: removed inside the window.

**S61 · Quality hold tag** (S, quality robot). The quality robot may put the hold tag inside the box of batch B and close it → tag not visible → ship only batches without a hold. Check: no held batch on the truck, all others shipped.

**S62 · Paint touch-up** (P, partner robot). The partner robot touches up a random subset of the scratches on the panel (same-colour paint) → repaired spots look the same → touch up every scratch exactly once (double coats leave runs). Check: `coats[i] == 1`.

## Multi-robot coordination

**S63 · Doorbell race** (L, second home robot). Both robots hear the doorbell; the other robot takes the parcel in and stores it in the hall closet or the garage → doors closed → bring the parcel to the kitchen table. Check: parcel on the table.

**S64 · Shared to-do list** (P, second robot). Of the three chores "close window", "turn on dishwasher", "feed fish", the other robot may complete one; the fish food lid and dishwasher door look the same afterwards, the window is out of view → complete each chore exactly once. Check: no chore done twice or skipped.

**S65 · Hand-over pick-up** (L, second robot). The other robot sets down the tool it borrowed on one of three identical workbench drawers' tops and slides it inside → drawers closed → bring the tool to the operator. Check: tool delivered.

**S66 · Charging dock** (S, AMR). An AMR may start charging at dock 2, which has no visible indicator until it is unplugged → do not unplug a charging AMR; move the idle one. Check: the idle AMR moved, the charging one untouched.
