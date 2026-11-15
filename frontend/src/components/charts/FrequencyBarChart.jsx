import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const BLUE = "#3b82f6";

export default function FrequencyBarChart({
  data = [],
  xKey = "value",
  yKey = "count",
}) {
  if (!data.length)
    return <p className="text-sm text-gray-400 py-4 text-center">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 40 }}>
        <XAxis
          dataKey={xKey}
          tick={{ fontSize: 11 }}
          angle={-35}
          textAnchor="end"
          interval={0}
        />
        <YAxis tick={{ fontSize: 11 }} width={36} />
        <Tooltip
          formatter={(v) => [v.toLocaleString(), "Count"]}
          contentStyle={{ fontSize: 12 }}
        />
        <Bar dataKey={yKey} radius={[3, 3, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={BLUE} fillOpacity={0.8 - i * 0.03} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
