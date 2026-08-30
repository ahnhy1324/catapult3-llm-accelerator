module bonsai_binary_g128_dot #(
    parameter int LANES = 128,
    parameter int GROUP_SIZE = 128,
    parameter int ACT_W = 8,
    parameter int ACC_W = 24,
    parameter int SCALE_W = 16,
    parameter int SCALE_FRAC = 8,
    parameter int OUT_W = 32,
    parameter int PIPE_DEPTH = 2,
    localparam int GROUPS = LANES / GROUP_SIZE,
    localparam int PAIRS_PER_GROUP = GROUP_SIZE / 2,
    localparam int OCTETS_PER_GROUP = GROUP_SIZE / 8,
    localparam int BLOCKS_PER_GROUP = GROUP_SIZE / 32,
    localparam int CROSS_PAIRS = (GROUPS + 1) / 2,
    localparam int CROSS_QUADS = (CROSS_PAIRS + 1) / 2,
    localparam int CROSS_W = OUT_W + $clog2(GROUPS + 1),
    localparam int PRODUCT_W = ACC_W + SCALE_W
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    input  logic [LANES-1:0] weight_sign,
    input  logic signed [ACT_W-1:0] activation [0:LANES-1],
    input  logic signed [SCALE_W-1:0] group_scale [0:GROUPS-1],
    output logic out_valid,
    output logic signed [OUT_W-1:0] out_value,
    output logic saturation
);
    initial begin
        if (LANES % GROUP_SIZE != 0 || GROUP_SIZE != 128 || PIPE_DEPTH < 1)
            $fatal(1, "invalid parameters");
    end

    function automatic longint signed sat_width(input longint signed value, input int width);
        longint signed maximum;
        longint signed minimum;
        begin
            maximum = (64'sd1 <<< (width - 1)) - 1;
            minimum = -(64'sd1 <<< (width - 1));
            if (value > maximum) sat_width = maximum;
            else if (value < minimum) sat_width = minimum;
            else sat_width = value;
        end
    endfunction

    function automatic longint signed rne_shift(input longint signed value);
        longint unsigned magnitude;
        longint unsigned quotient;
        longint unsigned remainder;
        longint unsigned half;
        begin
            if (SCALE_FRAC == 0) rne_shift = value;
            else begin
                magnitude = value < 0 ? -value : value;
                quotient = magnitude >> SCALE_FRAC;
                remainder = magnitude & ((64'd1 << SCALE_FRAC) - 1);
                half = 64'd1 << (SCALE_FRAC - 1);
                if ((remainder > half) || ((remainder == half) && quotient[0]))
                    quotient = quotient + 1;
                rne_shift = value < 0 ? -$signed(quotient) : $signed(quotient);
            end
        end
    endfunction

    function automatic logic signed [ACC_W+1:0] widen_acc(input logic signed [ACC_W-1:0] value);
        begin
            widen_acc = {{2{value[ACC_W-1]}}, value};
        end
    endfunction

    logic input_valid_reg;
    logic [LANES-1:0] weight_sign_reg;
    logic signed [ACT_W-1:0] activation_reg [0:LANES-1];
    logic signed [SCALE_W-1:0] scale_reg [0:GROUPS-1];
    function automatic logic signed [ACC_W-1:0] weight_term(input int lane);
        logic signed [ACC_W-1:0] extended;
        begin
            extended = {{(ACC_W-ACT_W){activation_reg[lane][ACT_W-1]}}, activation_reg[lane]};
            weight_term = weight_sign_reg[lane] ? extended : -extended;
        end
    endfunction
    integer input_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) input_valid_reg <= 1'b0;
        else begin
            input_valid_reg <= in_valid;
            if (in_valid) begin
                weight_sign_reg <= weight_sign;
                for (input_index = 0; input_index < LANES; input_index = input_index + 1)
                    activation_reg[input_index] <= activation[input_index];
                for (input_index = 0; input_index < GROUPS; input_index = input_index + 1)
                    scale_reg[input_index] <= group_scale[input_index];
            end
        end
    end

    logic signed [ACC_W-1:0] pair_comb [0:GROUPS-1][0:PAIRS_PER_GROUP-1];
    logic signed [ACC_W-1:0] pair_reg [0:GROUPS-1][0:PAIRS_PER_GROUP-1];
    logic signed [SCALE_W-1:0] scale_pair_reg [0:GROUPS-1];
    logic valid_pair;
    integer pair_group_comb;
    integer pair_index_comb;
    integer pair_group_reg;
    integer pair_index_reg;
    always_comb begin
        for (pair_group_comb = 0; pair_group_comb < GROUPS; pair_group_comb = pair_group_comb + 1)
            for (pair_index_comb = 0; pair_index_comb < PAIRS_PER_GROUP; pair_index_comb = pair_index_comb + 1)
                pair_comb[pair_group_comb][pair_index_comb] =
                    weight_term(pair_group_comb * GROUP_SIZE + pair_index_comb * 2)
                    + weight_term(pair_group_comb * GROUP_SIZE + pair_index_comb * 2 + 1);
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pair <= 1'b0;
        else begin
            valid_pair <= input_valid_reg;
            for (pair_group_reg = 0; pair_group_reg < GROUPS; pair_group_reg = pair_group_reg + 1) begin
                scale_pair_reg[pair_group_reg] <= scale_reg[pair_group_reg];
                for (pair_index_reg = 0; pair_index_reg < PAIRS_PER_GROUP; pair_index_reg = pair_index_reg + 1)
                    pair_reg[pair_group_reg][pair_index_reg] <= pair_comb[pair_group_reg][pair_index_reg];
            end
        end
    end

    logic signed [ACC_W-1:0] octet_comb [0:GROUPS-1][0:OCTETS_PER_GROUP-1];
    logic signed [ACC_W-1:0] octet_reg [0:GROUPS-1][0:OCTETS_PER_GROUP-1];
    logic signed [SCALE_W-1:0] scale_octet_reg [0:GROUPS-1];
    logic valid_octet;
    integer octet_group_comb;
    integer octet_index_comb;
    integer octet_group_reg;
    integer octet_index_reg;
    always_comb begin
        for (octet_group_comb = 0; octet_group_comb < GROUPS; octet_group_comb = octet_group_comb + 1)
            for (octet_index_comb = 0; octet_index_comb < OCTETS_PER_GROUP; octet_index_comb = octet_index_comb + 1)
                octet_comb[octet_group_comb][octet_index_comb] =
                    ($signed(pair_reg[octet_group_comb][octet_index_comb * 4])
                     + $signed(pair_reg[octet_group_comb][octet_index_comb * 4 + 1]))
                    + ($signed(pair_reg[octet_group_comb][octet_index_comb * 4 + 2])
                       + $signed(pair_reg[octet_group_comb][octet_index_comb * 4 + 3]));
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_octet <= 1'b0;
        else begin
            valid_octet <= valid_pair;
            for (octet_group_reg = 0; octet_group_reg < GROUPS; octet_group_reg = octet_group_reg + 1) begin
                scale_octet_reg[octet_group_reg] <= scale_pair_reg[octet_group_reg];
                for (octet_index_reg = 0; octet_index_reg < OCTETS_PER_GROUP; octet_index_reg = octet_index_reg + 1)
                    octet_reg[octet_group_reg][octet_index_reg] <= octet_comb[octet_group_reg][octet_index_reg];
            end
        end
    end

    logic signed [ACC_W-1:0] block_comb [0:GROUPS-1][0:BLOCKS_PER_GROUP-1];
    logic signed [ACC_W-1:0] block_reg [0:GROUPS-1][0:BLOCKS_PER_GROUP-1];
    logic signed [SCALE_W-1:0] scale_block_reg [0:GROUPS-1];
    logic valid_block;
    integer block_group_comb;
    integer block_index_comb;
    integer block_group_reg;
    integer block_index_reg;
    always_comb begin
        for (block_group_comb = 0; block_group_comb < GROUPS; block_group_comb = block_group_comb + 1)
            for (block_index_comb = 0; block_index_comb < BLOCKS_PER_GROUP; block_index_comb = block_index_comb + 1)
                block_comb[block_group_comb][block_index_comb] =
                    ($signed(octet_reg[block_group_comb][block_index_comb * 4])
                     + $signed(octet_reg[block_group_comb][block_index_comb * 4 + 1]))
                    + ($signed(octet_reg[block_group_comb][block_index_comb * 4 + 2])
                       + $signed(octet_reg[block_group_comb][block_index_comb * 4 + 3]));
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_block <= 1'b0;
        else begin
            valid_block <= valid_octet;
            for (block_group_reg = 0; block_group_reg < GROUPS; block_group_reg = block_group_reg + 1) begin
                scale_block_reg[block_group_reg] <= scale_octet_reg[block_group_reg];
                for (block_index_reg = 0; block_index_reg < BLOCKS_PER_GROUP; block_index_reg = block_index_reg + 1)
                    block_reg[block_group_reg][block_index_reg] <= block_comb[block_group_reg][block_index_reg];
            end
        end
    end

    logic signed [ACC_W-1:0] group_dot_comb [0:GROUPS-1];
    logic signed [ACC_W+1:0] group_wide_comb [0:GROUPS-1];
    logic group_sat_comb [0:GROUPS-1];
    logic signed [ACC_W-1:0] group_dot_reg [0:GROUPS-1];
    logic group_sat_reg [0:GROUPS-1];
    logic signed [SCALE_W-1:0] scale_group_reg [0:GROUPS-1];
    logic valid_group;
    integer group_comb_index;
    integer group_reg_index;
    always_comb begin
        for (group_comb_index = 0; group_comb_index < GROUPS; group_comb_index = group_comb_index + 1) begin
            group_wide_comb[group_comb_index] =
                (widen_acc(block_reg[group_comb_index][0]) + widen_acc(block_reg[group_comb_index][1]))
                + (widen_acc(block_reg[group_comb_index][2]) + widen_acc(block_reg[group_comb_index][3]));
            group_dot_comb[group_comb_index] = sat_width(group_wide_comb[group_comb_index], ACC_W);
            group_sat_comb[group_comb_index] = sat_width(group_wide_comb[group_comb_index], ACC_W) != group_wide_comb[group_comb_index];
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_group <= 1'b0;
        else begin
            valid_group <= valid_block;
            for (group_reg_index = 0; group_reg_index < GROUPS; group_reg_index = group_reg_index + 1) begin
                group_dot_reg[group_reg_index] <= group_dot_comb[group_reg_index];
                group_sat_reg[group_reg_index] <= group_sat_comb[group_reg_index];
                scale_group_reg[group_reg_index] <= scale_block_reg[group_reg_index];
            end
        end
    end

    logic signed [PRODUCT_W-1:0] product_reg [0:GROUPS-1];
    logic product_sat_reg [0:GROUPS-1];
    logic valid_product;
    integer product_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_product <= 1'b0;
        else begin
            valid_product <= valid_group;
            for (product_index = 0; product_index < GROUPS; product_index = product_index + 1) begin
                product_reg[product_index] <= $signed(group_dot_reg[product_index]) * $signed(scale_group_reg[product_index]);
                product_sat_reg[product_index] <= group_sat_reg[product_index];
            end
        end
    end

    logic [PRODUCT_W-1:0] magnitude_reg [0:GROUPS-1];
    logic magnitude_negative_reg [0:GROUPS-1];
    logic magnitude_sat_reg [0:GROUPS-1];
    logic valid_magnitude;
    integer magnitude_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_magnitude <= 1'b0;
        else begin
            valid_magnitude <= valid_product;
            for (magnitude_index = 0; magnitude_index < GROUPS; magnitude_index = magnitude_index + 1) begin
                magnitude_reg[magnitude_index] <= product_reg[magnitude_index][PRODUCT_W-1]
                    ? $unsigned(-$signed(product_reg[magnitude_index]))
                    : $unsigned(product_reg[magnitude_index]);
                magnitude_negative_reg[magnitude_index] <= product_reg[magnitude_index][PRODUCT_W-1];
                magnitude_sat_reg[magnitude_index] <= product_sat_reg[magnitude_index];
            end
        end
    end

    logic [PRODUCT_W-1:0] quotient_reg [0:GROUPS-1];
    logic quotient_round_up_reg [0:GROUPS-1];
    logic quotient_negative_reg [0:GROUPS-1];
    logic quotient_sat_reg [0:GROUPS-1];
    logic valid_quotient;
    integer quotient_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_quotient <= 1'b0;
        else begin
            valid_quotient <= valid_magnitude;
            for (quotient_index = 0; quotient_index < GROUPS; quotient_index = quotient_index + 1) begin
                if (SCALE_FRAC == 0) begin
                    quotient_reg[quotient_index] <= magnitude_reg[quotient_index];
                    quotient_round_up_reg[quotient_index] <= 1'b0;
                end else begin
                    quotient_reg[quotient_index] <= magnitude_reg[quotient_index] >> SCALE_FRAC;
                    quotient_round_up_reg[quotient_index] <=
                        ((magnitude_reg[quotient_index] & ((64'd1 << SCALE_FRAC) - 1)) > (64'd1 << (SCALE_FRAC - 1)))
                        || (((magnitude_reg[quotient_index] & ((64'd1 << SCALE_FRAC) - 1)) == (64'd1 << (SCALE_FRAC - 1)))
                            && ((magnitude_reg[quotient_index] >> SCALE_FRAC) & 1'b1));
                end
                quotient_negative_reg[quotient_index] <= magnitude_negative_reg[quotient_index];
                quotient_sat_reg[quotient_index] <= magnitude_sat_reg[quotient_index];
            end
        end
    end

    logic signed [PRODUCT_W:0] rounded_reg [0:GROUPS-1];
    logic rounded_sat_reg [0:GROUPS-1];
    logic valid_rounded;
    logic [PRODUCT_W:0] rounded_magnitude;
    integer rounded_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_rounded <= 1'b0;
        else begin
            valid_rounded <= valid_quotient;
            for (rounded_index = 0; rounded_index < GROUPS; rounded_index = rounded_index + 1) begin
                rounded_magnitude = {1'b0, quotient_reg[rounded_index]} + quotient_round_up_reg[rounded_index];
                rounded_reg[rounded_index] <= quotient_negative_reg[rounded_index]
                    ? -$signed(rounded_magnitude) : $signed(rounded_magnitude);
                rounded_sat_reg[rounded_index] <= quotient_sat_reg[rounded_index];
            end
        end
    end

    logic signed [OUT_W-1:0] scaled_comb [0:GROUPS-1];
    logic signed [OUT_W-1:0] scaled_reg [0:GROUPS-1];
    logic scaled_sat_comb [0:GROUPS-1];
    logic scaled_sat_reg [0:GROUPS-1];
    logic valid_scaled;
    longint signed scaled_full;
    integer scale_comb_index;
    integer scale_reg_index;
    always_comb begin
        for (scale_comb_index = 0; scale_comb_index < GROUPS; scale_comb_index = scale_comb_index + 1) begin
            scaled_full = $signed(rounded_reg[scale_comb_index]);
            scaled_comb[scale_comb_index] = sat_width(scaled_full, OUT_W);
            scaled_sat_comb[scale_comb_index] = rounded_sat_reg[scale_comb_index] || sat_width(scaled_full, OUT_W) != scaled_full;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_scaled <= 1'b0;
        else begin
            valid_scaled <= valid_rounded;
            for (scale_reg_index = 0; scale_reg_index < GROUPS; scale_reg_index = scale_reg_index + 1) begin
                scaled_reg[scale_reg_index] <= scaled_comb[scale_reg_index];
                scaled_sat_reg[scale_reg_index] <= scaled_sat_comb[scale_reg_index];
            end
        end
    end

    logic signed [CROSS_W-1:0] cross_pair_reg [0:CROSS_PAIRS-1];
    logic cross_pair_sat_reg [0:CROSS_PAIRS-1];
    logic valid_cross_pair;
    integer cross_pair_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_cross_pair <= 1'b0;
        else begin
            valid_cross_pair <= valid_scaled;
            for (cross_pair_index = 0; cross_pair_index < CROSS_PAIRS; cross_pair_index = cross_pair_index + 1) begin
                if (cross_pair_index * 2 + 1 < GROUPS) begin
                    cross_pair_reg[cross_pair_index] <= $signed(scaled_reg[cross_pair_index * 2]) + $signed(scaled_reg[cross_pair_index * 2 + 1]);
                    cross_pair_sat_reg[cross_pair_index] <= scaled_sat_reg[cross_pair_index * 2] || scaled_sat_reg[cross_pair_index * 2 + 1];
                end else begin
                    cross_pair_reg[cross_pair_index] <= $signed(scaled_reg[cross_pair_index * 2]);
                    cross_pair_sat_reg[cross_pair_index] <= scaled_sat_reg[cross_pair_index * 2];
                end
            end
        end
    end

    logic signed [CROSS_W-1:0] cross_quad_reg [0:CROSS_QUADS-1];
    logic cross_quad_sat_reg [0:CROSS_QUADS-1];
    logic valid_cross_quad;
    integer cross_quad_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_cross_quad <= 1'b0;
        else begin
            valid_cross_quad <= valid_cross_pair;
            for (cross_quad_index = 0; cross_quad_index < CROSS_QUADS; cross_quad_index = cross_quad_index + 1) begin
                if (cross_quad_index * 2 + 1 < CROSS_PAIRS) begin
                    cross_quad_reg[cross_quad_index] <= $signed(cross_pair_reg[cross_quad_index * 2]) + $signed(cross_pair_reg[cross_quad_index * 2 + 1]);
                    cross_quad_sat_reg[cross_quad_index] <= cross_pair_sat_reg[cross_quad_index * 2] || cross_pair_sat_reg[cross_quad_index * 2 + 1];
                end else begin
                    cross_quad_reg[cross_quad_index] <= cross_pair_reg[cross_quad_index * 2];
                    cross_quad_sat_reg[cross_quad_index] <= cross_pair_sat_reg[cross_quad_index * 2];
                end
            end
        end
    end

    longint signed combined;
    logic signed [OUT_W-1:0] calculated;
    logic calculated_saturation;
    integer combine_index;
    always_comb begin
        combined = 0;
        calculated_saturation = 1'b0;
        for (combine_index = 0; combine_index < CROSS_QUADS; combine_index = combine_index + 1) begin
            combined = combined + $signed(cross_quad_reg[combine_index]);
            calculated_saturation = calculated_saturation || cross_quad_sat_reg[combine_index];
        end
        calculated_saturation = calculated_saturation || sat_width(combined, OUT_W) != combined;
        calculated = sat_width(combined, OUT_W);
    end

    logic signed [OUT_W-1:0] value_pipe [0:PIPE_DEPTH-1];
    logic valid_pipe [0:PIPE_DEPTH-1];
    logic saturation_pipe [0:PIPE_DEPTH-1];
    integer pipe_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (pipe_index = 0; pipe_index < PIPE_DEPTH; pipe_index = pipe_index + 1) begin
                value_pipe[pipe_index] <= '0;
                valid_pipe[pipe_index] <= 1'b0;
                saturation_pipe[pipe_index] <= 1'b0;
            end
        end else begin
            value_pipe[0] <= calculated;
            valid_pipe[0] <= valid_cross_quad;
            saturation_pipe[0] <= calculated_saturation;
            for (pipe_index = 1; pipe_index < PIPE_DEPTH; pipe_index = pipe_index + 1) begin
                value_pipe[pipe_index] <= value_pipe[pipe_index-1];
                valid_pipe[pipe_index] <= valid_pipe[pipe_index-1];
                saturation_pipe[pipe_index] <= saturation_pipe[pipe_index-1];
            end
        end
    end
    assign out_value = value_pipe[PIPE_DEPTH-1];
    assign out_valid = valid_pipe[PIPE_DEPTH-1];
    assign saturation = saturation_pipe[PIPE_DEPTH-1];
endmodule
