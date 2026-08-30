module tb_microbench;
    timeunit 1ns;
    timeprecision 1ps;
    `include "generated_vectors.svh"

    logic clk = 1'b0;
    always #2.5 clk = ~clk;
    logic rst_n = 1'b0;

    logic binary_valid;
    // Icarus collapses a one-element unpacked array port during elaboration.
    // Use two g128 groups and zero the second group; this also exercises the
    // cross-group pipeline while preserving the 128-lane golden result.
    logic [255:0] binary_weight;
    logic signed [7:0] binary_activation [0:255];
    logic signed [15:0] binary_scale [0:1];
    logic binary_out_valid;
    logic signed [15:0] binary_out;
    logic binary_saturation;
    bonsai_binary_g128_dot #(
        .LANES(256), .GROUP_SIZE(128), .ACT_W(8), .ACC_W(14),
        .SCALE_W(16), .SCALE_FRAC(1), .OUT_W(16), .PIPE_DEPTH(2)
    ) binary_dut (
        .clk, .rst_n, .in_valid(binary_valid), .weight_sign(binary_weight),
        .activation(binary_activation), .group_scale(binary_scale),
        .out_valid(binary_out_valid), .out_value(binary_out), .saturation(binary_saturation)
    );

    logic direct_valid;
    logic [7:0] direct_packed [0:1];
    logic signed [7:0] ternary_activation [0:9];
    logic direct_out_valid;
    logic signed [15:0] direct_out;
    logic direct_saturation;
    bitnet_direct_ternary #(.LANES(10), .OUT_W(16), .PIPE_DEPTH(2)) direct_dut (
        .clk, .rst_n, .in_valid(direct_valid), .packed_weight(direct_packed),
        .activation(ternary_activation), .out_valid(direct_out_valid),
        .out_value(direct_out), .saturation(direct_saturation)
    );

    logic tl5_build_start;
    logic tl5_ready;
    logic tl5_valid;
    logic [7:0] tl5_packed [0:1];
    logic tl5_out_valid;
    logic signed [15:0] tl5_out;
    logic tl5_saturation;
    bitnet_tl5 #(.LANES(10), .OUT_W(16), .PIPE_DEPTH(2)) tl5_dut (
        .clk, .rst_n, .build_start(tl5_build_start), .activation(ternary_activation),
        .table_ready(tl5_ready), .in_valid(tl5_valid), .packed_weight(tl5_packed),
        .out_valid(tl5_out_valid), .out_value(tl5_out), .saturation(tl5_saturation)
    );

    integer binary_seen = 0;
    integer direct_seen = 0;
    integer tl5_seen = 0;
    logic binary_stream_started = 1'b0;
    logic direct_stream_started = 1'b0;
    logic tl5_stream_started = 1'b0;
    integer tl5_build_cycles = 0;
    logic tl5_build_counting = 1'b0;
    always @(posedge clk) begin
        #1;
        if (binary_stream_started && binary_seen < BINARY_VECTORS && !binary_out_valid)
            $fatal(1, "binary output bubble after stream start");
        if (direct_stream_started && direct_seen < TERNARY_VECTORS && !direct_out_valid)
            $fatal(1, "direct output bubble after stream start");
        if (tl5_stream_started && tl5_seen < TERNARY_VECTORS && !tl5_out_valid)
            $fatal(1, "TL5 output bubble after stream start");
        if (binary_out_valid) begin
            binary_stream_started = 1'b1;
            if ($signed(binary_out) !== binary_expected_vector[binary_seen])
                $fatal(1, "binary mismatch index=%0d got=%0d expected=%0d", binary_seen, $signed(binary_out), binary_expected_vector[binary_seen]);
            if (binary_saturation !== binary_saturation_vector[binary_seen])
                $fatal(1, "binary saturation mismatch index=%0d", binary_seen);
            binary_seen = binary_seen + 1;
        end
        if (direct_out_valid) begin
            direct_stream_started = 1'b1;
            if ($signed(direct_out) !== ternary_expected_vector[direct_seen])
                $fatal(1, "direct mismatch index=%0d got=%0d expected=%0d", direct_seen, $signed(direct_out), ternary_expected_vector[direct_seen]);
            direct_seen = direct_seen + 1;
        end
        if (tl5_out_valid) begin
            tl5_stream_started = 1'b1;
            if ($signed(tl5_out) !== ternary_expected_vector[tl5_seen])
                $fatal(1, "TL5 mismatch index=%0d got=%0d expected=%0d", tl5_seen, $signed(tl5_out), ternary_expected_vector[tl5_seen]);
            tl5_seen = tl5_seen + 1;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            tl5_build_cycles = 0;
            tl5_build_counting = 1'b0;
        end else if (tl5_build_start) begin
            tl5_build_cycles = 0;
            tl5_build_counting = 1'b1;
        end else if (tl5_build_counting && !tl5_ready) begin
            tl5_build_cycles = tl5_build_cycles + 1;
        end else if (tl5_build_counting && tl5_ready) begin
            tl5_build_counting = 1'b0;
        end
    end

    integer vector_index;
    integer lane_index;
    initial begin
        binary_valid = 0;
        direct_valid = 0;
        tl5_valid = 0;
        tl5_build_start = 0;
        repeat (3) @(negedge clk);
        rst_n = 1;

        for (lane_index = 0; lane_index < 10; lane_index = lane_index + 1)
            ternary_activation[lane_index] = ternary_activation_vector[lane_index];
        @(negedge clk);
        tl5_build_start = 1;
        @(negedge clk);
        tl5_build_start = 0;

        for (vector_index = 0; vector_index < BINARY_VECTORS; vector_index = vector_index + 1) begin
            @(negedge clk);
            binary_valid = 1;
            binary_scale[0] = binary_scale_vector[vector_index];
            binary_scale[1] = '0;
            for (lane_index = 0; lane_index < 256; lane_index = lane_index + 1) begin
                if (lane_index < 128) begin
                    binary_activation[lane_index] = binary_activation_vector[vector_index][lane_index];
                    binary_weight[lane_index] = binary_sign_vector[vector_index][lane_index] > 0;
                end else begin
                    binary_activation[lane_index] = '0;
                    binary_weight[lane_index] = 1'b0;
                end
            end
        end
        @(negedge clk);
        binary_valid = 0;

        for (vector_index = 0; vector_index < TERNARY_VECTORS; vector_index = vector_index + 1) begin
            @(negedge clk);
            direct_valid = 1;
            direct_packed[0] = ternary_packed_vector[vector_index][0];
            direct_packed[1] = ternary_packed_vector[vector_index][1];
        end
        @(negedge clk);
        direct_valid = 0;

        wait (tl5_ready);
        for (vector_index = 0; vector_index < TERNARY_VECTORS; vector_index = vector_index + 1) begin
            @(negedge clk);
            tl5_valid = 1;
            tl5_packed[0] = ternary_packed_vector[vector_index][0];
            tl5_packed[1] = ternary_packed_vector[vector_index][1];
        end
        @(negedge clk);
        tl5_valid = 0;
        wait (binary_seen == BINARY_VECTORS && direct_seen == TERNARY_VECTORS && tl5_seen == TERNARY_VECTORS);
        repeat (2) @(posedge clk);
        if (tl5_build_cycles != 244)
            $fatal(1, "TL5 table build latency got=%0d expected=244", tl5_build_cycles);
        $display("MICROBENCH_PASS binary=%0d direct=%0d tl5=%0d table_build_cycles=%0d II=1", binary_seen, direct_seen, tl5_seen, tl5_build_cycles);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout");
    end
endmodule
