//
//  bkout.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//
// Standard includes
#include <stdbool.h>
#include <stdint.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>


//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------
#define FIXED_BITS = 15
//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum{
  USE_XYP,
  USE_PYX
} key_format_t;

typedef enum
{
  REGION_SYSTEM,
  REGION_CONVOLUTION,
  REGION_PROVENANCE,
} region_t;


//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------
uint32_t time;
static int32_t width;
static int32_t height;
static int32_t out_width;
static int32_t out_height;
static int32_t kernel_width;
static int32_t kernel_height;
static int32_t half_kernel_width;
static int32_t half_kernel_height;
static int32_t kernel_size;
static int32_t step_width;
static int32_t step_height;
static int32_t *kernel = NULL;
static uint32_t time_window_length;
static int32_t threshold;
static uint8_t ***sampling = NULL;
static uint32_t **last_times = NULL;
static uint8_t width_bits;
static uint8_t height_bits;
static uint8_t polarity_bits;
static uint8_t out_width_bits;
static uint8_t out_height_bits;

static uint32_t width_mask;
static uint32_t height_mask;
static uint32_t polarity_mask;
static uint32_t out_width_mask;
static uint32_t out_height_mask;

static int32_t polarity;
static uint8_t use_xyp_or_pyx;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

//! the number of timer ticks that this model should run for before exiting.
static uint32_t simulation_ticks = 0;


//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
// polarity (p), row (y), column (x)
static inline void extract_pyx(
            uint32_t in_key, int32_t *x, int32_t *y, int32_t *p,
            uint8_t width_bits, uint8_t height_bits, uint8_t polarity_bits,
            uint32_t width_mask, uint32_t height_mask, uint32_t polarity_mask){
  use(polarity_bits);
  *p = (in_key >> (width_bits + height_bits)) & polarity_mask;
  *y = (in_key >> width_bits) & height_mask;
  *x = in_key & width_mask;
}


static inline void extract_xyp(
            uint32_t in_key, int32_t *x, int32_t *y, int32_t *p,
            uint8_t width_bits, uint8_t height_bits, uint8_t polarity_bits,
            uint32_t width_mask, uint32_t height_mask, uint32_t polarity_mask){
  use(width_bits);
  *x = (in_key >> (height_bits + polarity_bits)) & width_mask;
  *y = (in_key >> polarity_bits) & height_mask;
  *p = in_key & polarity_mask;
}


static inline uint32_t pack_pyx(uint32_t row, uint32_t col, uint32_t polarity,
            uint8_t width_bits, uint8_t height_bits, uint8_t polarity_bits,
            uint32_t width_mask, uint32_t height_mask, uint32_t polarity_mask){
  use(polarity_bits);
  uint32_t k = ((polarity & polarity_mask) << (width_bits + height_bits)) +
               ((row & height_mask) << width_bits ) +
               (col & width_mask);
  return k;
}


static inline uint32_t pack_xyp(uint32_t row, uint32_t col, uint32_t polarity,
            uint8_t width_bits, uint8_t height_bits, uint8_t polarity_bits,
            uint32_t width_mask, uint32_t height_mask, uint32_t polarity_mask){
  use(width_bits);
  uint32_t k = ((col & width_mask) << (height_bits + polarity_bits)) +
               ((row & height_mask) << polarity_bits) +
               (polarity & polarity_mask);
  return k;
}


static inline void send_spike_out(uint32_t row, uint32_t col, uint32_t polarity,
            uint8_t width_bits, uint8_t height_bits, uint8_t polarity_bits,
            uint32_t width_mask, uint32_t height_mask, uint32_t polarity_mask){

  uint32_t out_key = 0;
  if (use_xyp_or_pyx == USE_PYX){
    out_key = pack_pyx(row, col, polarity, width_bits, height_bits, polarity_bits,
                       width_mask, height_mask, polarity_mask);
  }else{
    out_key = pack_xyp(row, col, polarity, width_bits, height_bits, polarity_bits,
                       width_mask, height_mask, polarity_mask);
  }
  spin1_send_mc_packet(key | out_key, 0, NO_PAYLOAD);
  log_debug("Neuron (%u, %u, %u) spiked!", row, col, polarity);
}


//----------------------------------------------------------------------------
// Static functions
//----------------------------------------------------------------------------


static bool initialize(uint32_t *timer_period){

  log_info("Initialise Convolution core: started");

  // Get the address this core's DTCM data starts at from SRAM
  address_t address = data_specification_get_data_address();

  // Read the header
  if (!data_specification_read_header(address)){
      return false;
  }

  // Get the timing details and set up the simulation interface
  if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
        APPLICATION_NAME_HASH, timer_period, &simulation_ticks, &infinite_run, 1,
        data_specification_get_region(REGION_PROVENANCE, address))){
      return false;
  }

  // Read breakout region
  address_t conv_app_region = data_specification_get_region(REGION_CONVOLUTION, address);
  key = *conv_app_region++;
  width = *conv_app_region++;
  height = *conv_app_region++;
  out_width = *conv_app_region++;
  out_height = *conv_app_region++;
  kernel_width = *conv_app_region++;
  kernel_height = *conv_app_region++;
  step_width = *conv_app_region++;
  step_height = *conv_app_region++;
  half_kernel_width = kernel_width/2;
  half_kernel_height = kernel_height/2;


  width_bits = (uint8_t)(*conv_app_region++);
  height_bits = (uint8_t)(*conv_app_region++);
  polarity_bits = (uint8_t)(*conv_app_region++);
  out_width_bits = (uint8_t)(*conv_app_region++);
  out_height_bits = (uint8_t)(*conv_app_region++);

  kernel_size = kernel_width*kernel_height;
  threshold = *conv_app_region++;
  use_xyp_or_pyx = *conv_app_region++;
  time_window_length = *conv_app_region++;
  polarity = *conv_app_region++;

  width_mask = (1 << width_bits) - 1;
  height_mask = (1 << height_bits) - 1;
  polarity_mask = (1 << polarity_bits) - 1;
  out_width_mask = (1 << out_width_bits) - 1;
  out_height_mask = (1 << out_height_bits) - 1;

  kernel = (int32_t *)(spin1_malloc(kernel_size*(sizeof(int32_t))));
  if(kernel == NULL){
    log_error("Unable to allocate memory for kernel of shape %u, %u",
              kernel_width, kernel_height);
     return false;
  }

  for(int32_t r = 0; r < kernel_height; r++){
    for(int32_t c = 0; c < kernel_width; c++){
      kernel[r*kernel_width + c] = (int32_t)(*conv_app_region++);
    }
  }

//  sampling = (uint8_t ***)(spin1_malloc(height*(sizeof(uint8_t **))));
//  if (sampling == NULL)
//
//  for(uint32_t r = 0; r < height; r++){
//    sampling[r] = (uint8_t **)(spin1_malloc(width*(sizeof(uint8_t *))));
//    for(uint32_t c = 0; c < width; c++){
//      sampling[r][c] = (uint8_t *)(spin1_malloc(kernel_size*(sizeof(uint8_t))));
//      for(uint32_t kr = 0; kr < kernel_height; kr++){
//        for(uint32_t kc = 0; kc < kernel_width; kc++){
//          sampling[r][c][kr*kernel_width + kc] = 0;
//        }
//      }
//    }
//  }

  last_times = (uint32_t **)(spin1_malloc(height*(sizeof(uint32_t *))));
  if(last_times == NULL){
    log_error("Unable to allocate memory for last_times[0] of shape %u, %u",
               width, height);
     return false;
  }
  for(int32_t r = 0; r < height; r++){
    last_times[r] = (uint32_t *)(spin1_malloc(width*sizeof(uint32_t)));
    if(last_times[r] == NULL){
      log_error("Unable to allocate memory for last_times[%u] of shape %u, %u",
                r, width, height);
      return false;
    }
    for(int32_t c = 0; c < width; c++){
      last_times[r][c] = 0x1FFFF;
    }
  }

  log_info("\tKey=%08x", key);
  log_info("\tTimer period=%d", *timer_period);
  log_info("\tShapes (width, height): ");
  log_info("\t\tsource (%u, %u)", width, height);
  log_info("\t\toutput (%u, %u)", out_width, out_height);
  log_info("\t\tkernel (%u, %u)", kernel_width, kernel_height);
  log_info("\t\tsample (%u, %u)", 1 << step_width, 1 << step_height);

  log_info("\tKernel:\n");
  for(int32_t kr = 0; kr < kernel_height; kr++){
    io_printf(IO_BUF, "\t[\t");
    for(int32_t kc = 0; kc < kernel_width; kc++){
      io_printf(IO_BUF, "%k\t", kernel[kr*kernel_width + kc]);
    }
    io_printf(IO_BUF, "]\n");
  }
  io_printf(IO_BUF, "\n");

  log_info("\tUse XYP(%u) or PYX(%u):  %u", USE_XYP, USE_PYX, use_xyp_or_pyx);
  log_info("\tTime window length:  %u", time_window_length);
  log_info("\tPolarity: %u", polarity);
  log_info("\tThreshold: %k", threshold);

  log_info("Initialise: completed successfully");

  return true;
}

//----------------------------------------------------------------------------
// Callbacks
//----------------------------------------------------------------------------
// incoming SDP message
/*void process_sdp (uint m, uint port)
*{
    sdp_msg_t *msg = (sdp_msg_t *) m;

    io_printf (IO_BUF, "SDP len %d, port %d - %s\n", msg->length, port, msg->data);
    // Port 1 - key data
    if (port == 1) spin1_memcpy(&keystate, msg->data, 4);
    spin1_msg_free (msg);
    if (port == 7) spin1_exit (0);
}*/

void timer_callback(uint unused, uint dummy)
{
  time++;

  use(unused);
  use(dummy);


  if (infinite_run != TRUE && time >= simulation_ticks) {

    log_info("Completed a run");

    // Enter pause and resume state to avoid another tick
    simulation_handle_pause_resume(NULL);

    log_info("Exiting Convolution Core on timer.");
    // Subtract 1 from the time so this tick gets done again on the next
    // run
    time -= 1;
    return;
  }
//  log_info("time %u", time);
  // Otherwise
//  else
//  {
//  }
}

void mc_packet_received_callback(uint key, uint payload)
{
  use(payload);

  int32_t x, y, p;
  int32_t r, c, kr, kc;
  int32_t sums[kernel_height][kernel_width];
  if(use_xyp_or_pyx == USE_XYP){
    extract_xyp(key, &x, &y, &p, width_bits, height_bits, polarity_bits,
                width_mask, height_mask, polarity_mask);
  }else {
    extract_pyx(key, &x, &y, &p, width_bits, height_bits, polarity_bits,
                width_mask, height_mask, polarity_mask);
  }


  if(p == polarity){
//    log_info("x, y, p = %u, %u, %u", x, y, p);

    if(x >= width || y >= height){
      return;
    }

    for(kr = 0; kr < kernel_height; kr++){
        for(kc = 0; kc < kernel_width; kc++){
            sums[kr][kc] = 0;
//            log_info("sums[%u][%u] = %k", kr, kc, sums[kr][kc]);
        }
    }
    last_times[y][x] = time;
    for(kr = -half_kernel_height; kr < half_kernel_height; kr++){
      r = y + kr;
//      log_info("r %d", r);
      if(r < 0 || r >= height){
        continue;
      }

      for(kc = -half_kernel_width; kc < half_kernel_width; kc++){
          c = x + kc;
//          log_info("c %d", c);
          if(c < 0 || c >= height){
            continue;
          }

          const uint32_t tdiff = time - last_times[r][c];
//          log_info("time now %u, last %u, diff %u",
//                   time, last_times[r][c], tdiff);
          if(tdiff < time_window_length){
            kr += half_kernel_height;
            kc += half_kernel_width;
            sums[kr][kc] += kernel[kr*kernel_width + kc];
//            log_info("%u, %u\t\tsums[%u][%u] = %k", y, x, kr, kc, sums[kr][kc]);
          }
      }
    }
    for(kr = -half_kernel_height; kr < half_kernel_height; kr++){
      r = y + kr;
      if(r < 0){
        continue;
      }

      for(kc = -half_kernel_width; kc < half_kernel_width; kc++){
        c = x + kc;
        if(c < 0){
          continue;
        }
        kr += half_kernel_height;
        kc += half_kernel_width;
//        log_info("sums[%u][%u] = %k", kr, kc, sums[kr][kc]);
        if(sums[kr][kc] >= threshold){
          log_info("MC key %u, x %u, y %u, p %u", key, x, y, p);
          log_info("MC out r %u, c %u, p %u", r, c, p);
          send_spike_out((r>>step_height), (c>>step_width), p,
                         out_width_bits, out_height_bits, polarity_bits,
                         out_width_mask, out_height_mask, polarity_mask);
        }
      }
    }
  }


}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


void rte_handler (uint code)
{
  // Save code

  sark.vcpu->user0 = code;
  sark.vcpu->user1 = (uint) sark.sdram_buf;

  // Copy ITCM to SDRAM

  sark_word_cpy (sark.sdram_buf, (void *) ITCM_BASE, ITCM_SIZE);

  // Copy DTCM to SDRAM

  sark_word_cpy (sark.sdram_buf + ITCM_SIZE, (void *) DTCM_BASE, DTCM_SIZE);

  // Try to re-establish consistent SARK state

  sark_vic_init ();

  sark_vic_set ((vic_slot) sark_vec->sark_slot, CPU_INT, 1, sark_int_han);

  uint *stack = sark_vec->stack_top - sark_vec->svc_stack;

  stack = cpu_init_mode (stack, IMASK_ALL+MODE_IRQ, sark_vec->irq_stack);
  stack = cpu_init_mode (stack, IMASK_ALL+MODE_FIQ, sark_vec->fiq_stack);
  (void)  cpu_init_mode (stack, IMASK_ALL+MODE_SYS, 0);

  cpu_set_cpsr (MODE_SYS);

  // ... and sleep

  while (1)
    cpu_wfi ();
}

//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    log_error("Error in initialisation - exiting!");
    rt_error(RTE_SWERR);
    return;
  }

  // Set timer tick (in microseconds)
  spin1_set_timer_tick(timer_period);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  time = UINT32_MAX;

  simulation_run();
}
