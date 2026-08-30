module bitnet_direct_ternary #(
    parameter int LANES = 10,
    parameter int ACT_W = 8,
    parameter int OUT_W = 24,
    parameter int PIPE_DEPTH = 2,
    localparam int PACKED_BYTES = (LANES + 4) / 5
) (
    input  logic clk,
    input  logic rst_n,
    input  logic in_valid,
    input  logic [7:0] packed_weight [0:PACKED_BYTES-1],
    input  logic signed [ACT_W-1:0] activation [0:LANES-1],
    output logic out_valid,
    output logic signed [OUT_W-1:0] out_value,
    output logic saturation
);
    initial begin
        if (LANES < 1 || PACKED_BYTES > 256 || PIPE_DEPTH < 1)
            $fatal(1, "invalid parameters");
    end

    logic input_valid_reg;
    logic [7:0] packed_weight_reg [0:PACKED_BYTES-1];
    logic signed [ACT_W-1:0] activation_reg [0:LANES-1];
    integer input_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) input_valid_reg <= 1'b0;
        else begin
            input_valid_reg <= in_valid;
            if (in_valid) begin
                for (input_index = 0; input_index < PACKED_BYTES; input_index = input_index + 1)
                    packed_weight_reg[input_index] <= packed_weight[input_index];
                for (input_index = 0; input_index < LANES; input_index = input_index + 1)
                    activation_reg[input_index] <= activation[input_index];
            end
        end
    end

    function automatic logic signed [OUT_W-1:0] decode_group(input int group_number, input int packed_value);
        integer remainder;
        integer digit0;
        integer digit1;
        integer digit2;
        integer digit3;
        integer digit4;
        logic signed [OUT_W-1:0] term0;
        logic signed [OUT_W-1:0] term1;
        logic signed [OUT_W-1:0] term2;
        logic signed [OUT_W-1:0] term3;
        logic signed [OUT_W-1:0] term4;
        begin
            if (packed_value > 242) begin
                decode_group = 0;
            end else begin
                digit4 = packed_value >= 162 ? 2 : (packed_value >= 81 ? 1 : 0);
                remainder = packed_value - digit4 * 81;
                digit3 = remainder >= 54 ? 2 : (remainder >= 27 ? 1 : 0);
                remainder = remainder - digit3 * 27;
                digit2 = remainder >= 18 ? 2 : (remainder >= 9 ? 1 : 0);
                remainder = remainder - digit2 * 9;
                digit1 = remainder >= 6 ? 2 : (remainder >= 3 ? 1 : 0);
                digit0 = remainder - digit1 * 3;
                term0 = '0;
                term1 = '0;
                term2 = '0;
                term3 = '0;
                term4 = '0;
                if (group_number * 5 < LANES) begin
                    term0 = $signed(activation_reg[group_number * 5]);
                    if (digit0 == 0) term0 = -term0;
                    else if (digit0 == 1) term0 = '0;
                end
                if (group_number * 5 + 1 < LANES) begin
                    term1 = $signed(activation_reg[group_number * 5 + 1]);
                    if (digit1 == 0) term1 = -term1;
                    else if (digit1 == 1) term1 = '0;
                end
                if (group_number * 5 + 2 < LANES) begin
                    term2 = $signed(activation_reg[group_number * 5 + 2]);
                    if (digit2 == 0) term2 = -term2;
                    else if (digit2 == 1) term2 = '0;
                end
                if (group_number * 5 + 3 < LANES) begin
                    term3 = $signed(activation_reg[group_number * 5 + 3]);
                    if (digit3 == 0) term3 = -term3;
                    else if (digit3 == 1) term3 = '0;
                end
                if (group_number * 5 + 4 < LANES) begin
                    term4 = $signed(activation_reg[group_number * 5 + 4]);
                    if (digit4 == 0) term4 = -term4;
                    else if (digit4 == 1) term4 = '0;
                end
                decode_group = (term0 + term1) + (term2 + term3) + term4;
            end
        end
    endfunction

    logic signed [OUT_W-1:0] partial_comb [0:PACKED_BYTES-1];
    logic signed [OUT_W-1:0] partial_reg [0:PACKED_BYTES-1];
    logic invalid_comb [0:PACKED_BYTES-1];
    logic invalid_reg [0:PACKED_BYTES-1];
    logic valid_partial;
    integer decode_index_comb;
    integer decode_index_reg;
    always_comb begin
        for (decode_index_comb = 0; decode_index_comb < PACKED_BYTES; decode_index_comb = decode_index_comb + 1) begin
            partial_comb[decode_index_comb] = decode_group(decode_index_comb, packed_weight_reg[decode_index_comb]);
            invalid_comb[decode_index_comb] = packed_weight_reg[decode_index_comb] > 8'd242;
        end
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_partial <= 1'b0;
        else begin
            valid_partial <= input_valid_reg;
            for (decode_index_reg = 0; decode_index_reg < PACKED_BYTES; decode_index_reg = decode_index_reg + 1) begin
                partial_reg[decode_index_reg] <= partial_comb[decode_index_reg];
                invalid_reg[decode_index_reg] <= invalid_comb[decode_index_reg];
            end
        end
    end

    // A fixed eight-level binary tree covers up to 256 packed bytes (1,280
    // trits). Unused leaves are explicitly zero, preserving II=1 at any
    // supported lane count including 640 and 672.
    logic signed [OUT_W-1:0] tree_reg [0:7][0:127];
    logic tree_invalid_reg [0:7][0:127];
    logic valid_tree [0:7];
    integer tree0_index;
    integer tree1_index;
    integer tree2_index;
    integer tree3_index;
    integer tree4_index;
    integer tree5_index;
    integer tree6_index;
    integer tree7_index;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_tree[0] <= 1'b0;
            valid_tree[1] <= 1'b0;
            valid_tree[2] <= 1'b0;
            valid_tree[3] <= 1'b0;
            valid_tree[4] <= 1'b0;
            valid_tree[5] <= 1'b0;
            valid_tree[6] <= 1'b0;
            valid_tree[7] <= 1'b0;
        end else begin
            valid_tree[0] <= valid_partial;
            for (tree0_index = 0; tree0_index < 128; tree0_index = tree0_index + 1) begin
                if (tree0_index * 2 + 1 < PACKED_BYTES) begin
                    tree_reg[0][tree0_index] <= $signed(partial_reg[tree0_index * 2]) + $signed(partial_reg[tree0_index * 2 + 1]);
                    tree_invalid_reg[0][tree0_index] <= invalid_reg[tree0_index * 2] || invalid_reg[tree0_index * 2 + 1];
                end else if (tree0_index * 2 < PACKED_BYTES) begin
                    tree_reg[0][tree0_index] <= partial_reg[tree0_index * 2];
                    tree_invalid_reg[0][tree0_index] <= invalid_reg[tree0_index * 2];
                end else begin
                    tree_reg[0][tree0_index] <= '0;
                    tree_invalid_reg[0][tree0_index] <= 1'b0;
                end
            end
            valid_tree[1] <= valid_tree[0];
            for (tree1_index = 0; tree1_index < 64; tree1_index = tree1_index + 1) begin
                tree_reg[1][tree1_index] <= $signed(tree_reg[0][tree1_index * 2]) + $signed(tree_reg[0][tree1_index * 2 + 1]);
                tree_invalid_reg[1][tree1_index] <= tree_invalid_reg[0][tree1_index * 2] || tree_invalid_reg[0][tree1_index * 2 + 1];
            end
            valid_tree[2] <= valid_tree[1];
            for (tree2_index = 0; tree2_index < 32; tree2_index = tree2_index + 1) begin
                tree_reg[2][tree2_index] <= $signed(tree_reg[1][tree2_index * 2]) + $signed(tree_reg[1][tree2_index * 2 + 1]);
                tree_invalid_reg[2][tree2_index] <= tree_invalid_reg[1][tree2_index * 2] || tree_invalid_reg[1][tree2_index * 2 + 1];
            end
            valid_tree[3] <= valid_tree[2];
            for (tree3_index = 0; tree3_index < 16; tree3_index = tree3_index + 1) begin
                tree_reg[3][tree3_index] <= $signed(tree_reg[2][tree3_index * 2]) + $signed(tree_reg[2][tree3_index * 2 + 1]);
                tree_invalid_reg[3][tree3_index] <= tree_invalid_reg[2][tree3_index * 2] || tree_invalid_reg[2][tree3_index * 2 + 1];
            end
            valid_tree[4] <= valid_tree[3];
            for (tree4_index = 0; tree4_index < 8; tree4_index = tree4_index + 1) begin
                tree_reg[4][tree4_index] <= $signed(tree_reg[3][tree4_index * 2]) + $signed(tree_reg[3][tree4_index * 2 + 1]);
                tree_invalid_reg[4][tree4_index] <= tree_invalid_reg[3][tree4_index * 2] || tree_invalid_reg[3][tree4_index * 2 + 1];
            end
            valid_tree[5] <= valid_tree[4];
            for (tree5_index = 0; tree5_index < 4; tree5_index = tree5_index + 1) begin
                tree_reg[5][tree5_index] <= $signed(tree_reg[4][tree5_index * 2]) + $signed(tree_reg[4][tree5_index * 2 + 1]);
                tree_invalid_reg[5][tree5_index] <= tree_invalid_reg[4][tree5_index * 2] || tree_invalid_reg[4][tree5_index * 2 + 1];
            end
            valid_tree[6] <= valid_tree[5];
            for (tree6_index = 0; tree6_index < 2; tree6_index = tree6_index + 1) begin
                tree_reg[6][tree6_index] <= $signed(tree_reg[5][tree6_index * 2]) + $signed(tree_reg[5][tree6_index * 2 + 1]);
                tree_invalid_reg[6][tree6_index] <= tree_invalid_reg[5][tree6_index * 2] || tree_invalid_reg[5][tree6_index * 2 + 1];
            end
            valid_tree[7] <= valid_tree[6];
            for (tree7_index = 0; tree7_index < 1; tree7_index = tree7_index + 1) begin
                tree_reg[7][tree7_index] <= $signed(tree_reg[6][tree7_index * 2]) + $signed(tree_reg[6][tree7_index * 2 + 1]);
                tree_invalid_reg[7][tree7_index] <= tree_invalid_reg[6][tree7_index * 2] || tree_invalid_reg[6][tree7_index * 2 + 1];
            end
        end
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
            value_pipe[0] <= tree_reg[7][0];
            valid_pipe[0] <= valid_tree[7];
            saturation_pipe[0] <= tree_invalid_reg[7][0];
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
